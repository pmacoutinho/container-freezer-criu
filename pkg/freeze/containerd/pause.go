package containerd

import (
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"github.com/containerd/containerd"
	"github.com/containerd/containerd/cio"
	"github.com/containerd/containerd/namespaces"
	runctypes "github.com/containerd/containerd/runtime/linux/runctypes"
	"github.com/containerd/containerd/runtime/v2/runc/options"
	"google.golang.org/grpc"

	"knative.dev/container-freezer/pkg/freeze/common"
)

const (
	defaultContainerdAddress = "/var/run/containerd/containerd.sock"
	// fifoDir is a host-mounted directory used for task IO FIFOs after
	// CRIU restore. It must be the same path inside the daemon container
	// and on the host so the containerd shim can open the FIFOs.
	fifoDir = "/run/freezer-fifo"
	// checkpointBaseDir is the host directory where CRIU checkpoint images
	// are stored. Using a filesystem path instead of the containerd content
	// store avoids the rootfs diff computation, which fails on overlay
	// filesystems that do not support xattr (e.g. kernel <5.11). It must be
	// on persistent storage, not tmpfs: /run (and /tmp on the RSUs) is
	// RAM-backed, so images stored there would not free the memory that
	// checkpointing is meant to release.
	checkpointBaseDir = "/var/lib/freezer-checkpoints"
)

// criLogWriter formats container output in CRI log format
// (<timestamp> <stream> F <message>\n) and appends it to a shared log file.
type criLogWriter struct {
	f      *os.File
	stream string // "stdout" or "stderr"
	mu     *sync.Mutex
}

func (w *criLogWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	ts := time.Now().UTC().Format(time.RFC3339Nano)
	// Write as a full CRI log line.
	_, err := fmt.Fprintf(w.f, "%s %s F %s", ts, w.stream, p)
	if err != nil {
		return 0, err
	}
	if p[len(p)-1] != '\n' {
		w.f.Write([]byte("\n"))
	}
	return len(p), nil
}

// withCheckpointOpenTcp tells CRIU to handle open TCP connections (--tcp-established).
func withCheckpointOpenTcp() containerd.CheckpointTaskOpts {
	return func(r *containerd.CheckpointTaskInfo) error {
		if containerd.CheckRuntime(r.Runtime(), "io.containerd.runc") {
			if r.Options == nil {
				r.Options = &options.CheckpointOptions{}
			}
			opts, ok := r.Options.(*options.CheckpointOptions)
			if !ok {
				return errors.New("invalid v2 shim checkpoint options format")
			}
			opts.OpenTcp = true
		} else {
			if r.Options == nil {
				r.Options = &runctypes.CheckpointOptions{}
			}
			opts, ok := r.Options.(*runctypes.CheckpointOptions)
			if !ok {
				return errors.New("invalid v1 shim checkpoint options format")
			}
			opts.OpenTcp = true
		}
		return nil
	}
}

// NewContainerdProvider returns a CRI based on Containerd
func NewContainerdProvider() (*ContainerdCRI, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	address := os.Getenv("CONTAINERD_ADDRESS")
	if address == "" {
		address = defaultContainerdAddress
	}

	conn, err := grpc.DialContext(ctx, address, grpc.WithInsecure(), grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(1024*1024*16)), grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
		return (&net.Dialer{}).DialContext(ctx, "unix", addr)
	}))
	if err != nil {
		return nil, err
	}

	client, err := containerd.NewWithConn(conn)
	if err != nil {
		return nil, err
	}

	return &ContainerdCRI{conn: conn, ctrd: client}, nil
}

type ContainerdCRI struct {
	conn        *grpc.ClientConn
	ctrd        *containerd.Client
	checkpoints sync.Map // containerID -> containerd.Image
}

// List returns a list of all non queue-proxy container IDs in a given pod.
func (c *ContainerdCRI) List(ctx context.Context, podKey string) ([]string, error) {
	return common.List(ctx, c.conn, podKey)
}

// Pause creates a CRIU checkpoint of the container (saving full process state
// to disk) and then kills the process to free RAM. The checkpoint is stored
// in containerd's content store for later restore.
// On arm64, where the overlay filesystem lacks xattr support (kernel <5.11),
// checkpoint images are saved to a filesystem path instead, bypassing the
// rootfs diff that requires xattr.
// Flow: containerd → runc checkpoint → criu dump
func (c *ContainerdCRI) Pause(ctx context.Context, containerID string) error {
	nctx := namespaces.WithNamespace(ctx, "k8s.io")
	ctr, err := c.ctrd.LoadContainer(nctx, containerID)
	if err != nil {
		return fmt.Errorf("%s checkpoint failed: %v", containerID, err)
	}
	task, err := ctr.Task(nctx, nil)
	if err != nil {
		return fmt.Errorf("%s checkpoint failed: %v", containerID, err)
	}

	if runtime.GOARCH == "arm64" {
		return c.pauseImagePath(nctx, containerID, task)
	}
	return c.pauseContentStore(nctx, containerID, task)
}

// pauseContentStore checkpoints via the containerd content store (default).
func (c *ContainerdCRI) pauseContentStore(ctx context.Context, containerID string, task containerd.Task) error {
	checkpoint, err := task.Checkpoint(ctx, withCheckpointOpenTcp())
	if err != nil {
		return fmt.Errorf("%s checkpoint failed: %v", containerID, err)
	}

	c.checkpoints.Store(containerID, checkpoint)

	_, err = task.Delete(ctx, containerd.WithProcessKill)
	if err != nil {
		return fmt.Errorf("%s kill after checkpoint failed: %v", containerID, err)
	}

	return nil
}

// pauseImagePath checkpoints to a filesystem path, skipping the rootfs diff
// that fails on overlay filesystems without xattr support.
func (c *ContainerdCRI) pauseImagePath(ctx context.Context, containerID string, task containerd.Task) error {
	imgPath := filepath.Join(checkpointBaseDir, containerID)
	if err := os.MkdirAll(imgPath, 0700); err != nil {
		return fmt.Errorf("%s checkpoint failed to create dir: %v", containerID, err)
	}

	_, err := task.Checkpoint(ctx,
		withCheckpointOpenTcp(),
		containerd.WithCheckpointImagePath(imgPath),
	)
	if err != nil {
		return fmt.Errorf("%s checkpoint failed: %v", containerID, err)
	}

	c.checkpoints.Store(containerID, imgPath)

	_, err = task.Delete(ctx, containerd.WithProcessKill)
	if err != nil {
		return fmt.Errorf("%s kill after checkpoint failed: %v", containerID, err)
	}

	return nil
}

// Resume restores a container from its CRIU checkpoint, recreating the process
// from disk. RAM is re-allocated and execution resumes from the exact point
// where Pause was called.
// Flow: containerd → runc restore → criu restore
func (c *ContainerdCRI) Resume(ctx context.Context, containerID string) error {
	nctx := namespaces.WithNamespace(ctx, "k8s.io")

	val, ok := c.checkpoints.LoadAndDelete(containerID)
	if !ok {
		return fmt.Errorf("%s has no checkpoint to restore from", containerID)
	}

	ctr, err := c.ctrd.LoadContainer(nctx, containerID)
	if err != nil {
		return fmt.Errorf("%s restore failed: %v", containerID, err)
	}

	ioCreator := c.buildIOCreator(nctx, ctr)

	// Restore from content store image or filesystem path depending on
	// what Pause stored.
	var restoreOpt containerd.NewTaskOpts
	switch v := val.(type) {
	case containerd.Image:
		restoreOpt = containerd.WithTaskCheckpoint(v)
	case string:
		restoreOpt = containerd.WithRestoreImagePath(v)
		defer os.RemoveAll(v)
	default:
		return fmt.Errorf("%s has invalid checkpoint type %T", containerID, val)
	}

	task, err := ctr.NewTask(nctx, ioCreator, restoreOpt)
	if err != nil {
		return fmt.Errorf("%s restore failed: %v", containerID, err)
	}

	if err := task.Start(nctx); err != nil {
		return fmt.Errorf("%s start after restore failed: %v", containerID, err)
	}

	return nil
}

// buildIOCreator reconstructs the CRI log path from container labels and
// returns an IO creator that formats output in CRI log format so kubectl
// logs works. FIFOs are created in fifoDir (a host-mounted path) so the
// containerd shim on the host can access them.
func (c *ContainerdCRI) buildIOCreator(ctx context.Context, ctr containerd.Container) cio.Creator {
	labels, err := ctr.Labels(ctx)
	if err != nil {
		return cio.NullIO
	}
	podNs := labels["io.kubernetes.pod.namespace"]
	podName := labels["io.kubernetes.pod.name"]
	podUID := labels["io.kubernetes.pod.uid"]
	ctrName := labels["io.kubernetes.container.name"]
	if podNs == "" || podName == "" || podUID == "" || ctrName == "" {
		return cio.NullIO
	}
	logPath := fmt.Sprintf("/var/log/pods/%s_%s_%s/%s/0.log",
		podNs, podName, podUID, ctrName)
	f, err := os.OpenFile(logPath, os.O_WRONLY|os.O_APPEND|os.O_CREATE, 0640)
	if err != nil {
		return cio.NullIO
	}
	mu := &sync.Mutex{}
	return cio.NewCreator(
		cio.WithFIFODir(fifoDir),
		cio.WithStreams(nil,
			&criLogWriter{f: f, stream: "stdout", mu: mu},
			&criLogWriter{f: f, stream: "stderr", mu: mu},
		),
	)
}
