"""Experiment-owned lifetime for the local monitor, task owner and publisher."""
from pathlib import Path

from wmal.monitor.catalog import RunCatalog
from wmal.monitor.frame_buffer import LatestFrameBuffer
from wmal.monitor.publisher import FramePublisher
from wmal.monitor.server import MonitorServer


class MonitorRuntime:
    def __init__(self, run_path, runs_root='runs', port=8765, camera='workcell_overview', max_fps=5,
                 task_service=None, capabilities=None):
        catalog = RunCatalog(runs_root)
        run = Path(run_path).resolve()
        if not run.is_relative_to(catalog.root) or run.is_symlink() or not run.is_dir():
            raise ValueError('Monitor run directory must be inside --monitor-runs-root')
        self.task_service = task_service
        self.task_shutdown_timeout_s = 10.
        self.run_id = catalog._run_id(catalog._relative(run))
        buffer = LatestFrameBuffer(max_history=2)
        self.publisher = FramePublisher(buffer, self.run_id, camera, max_fps=max_fps)
        self.server = MonitorServer(catalog, buffer, port=port, task_service=task_service,
                                    capabilities=capabilities)
        self.server.register_source(self.run_id, self.publisher)
        if task_service is not None:
            self.server.task_run_id = self.run_id

    def attach_task_service(self, service, capabilities, *, timeout_s=10.):
        """Attach before serving; the owner closes before display resources."""
        if getattr(self, 'task_service', None) is not None:
            raise RuntimeError('A task service is already attached')
        self.task_service = service
        self.task_shutdown_timeout_s = timeout_s
        self.server.task_service = service
        self.server.task_run_id = self.run_id
        self.server.capabilities = capabilities

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.address[1]}/?run={self.run_id}'

    def __enter__(self):
        self.server.start()
        return self

    def __exit__(self, *_):
        service = getattr(self, 'task_service', None)
        if service is not None:
            stopped = service.close(timeout_s=self.task_shutdown_timeout_s)
            if not stopped:
                # Preserve publisher resources while a slow owner is still
                # unwinding physics; prevent any further browser submissions.
                self.server.shutdown()
                raise RuntimeError('Task owner did not finish within shutdown timeout')
        self.publisher.close()
        self.server.shutdown()
