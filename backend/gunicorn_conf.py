import os

# Number of worker processes for handling requests.
# This can be adjusted based on the server's CPU cores and expected load.
# Can be overridden by the GUNICORN_WORKERS environment variable.
workers = int(os.environ.get('GUNICORN_WORKERS', '2'))

# The type of worker that Gunicorn will use.
# UvicornWorker is used for ASGI applications like FastAPI.
# Can be overridden by the GUNICORN_WORKER_CLASS environment variable.
worker_class = os.environ.get('GUNICORN_WORKER_CLASS', 'uvicorn.workers.UvicornWorker')

# The socket to bind to.
# '0.0.0.0:8000' makes the application accessible externally on port 8000.
# Can be overridden by the GUNICORN_BIND environment variable.
bind = os.environ.get('GUNICORN_BIND', '0.0.0.0:8000')

# Logging configuration.
# '-' logs access and error messages to stdout and stderr, respectively.
# This is useful for containerized environments where logs are typically collected from stdout/stderr.
accesslog = '-'
errorlog = '-'

# The granularity of log output.
# Options include 'debug', 'info', 'warning', 'error', 'critical'.
# Can be overridden by the GUNICORN_LOGLEVEL environment variable.
loglevel = os.environ.get('GUNICORN_LOGLEVEL', 'info')

# Worker timeout in seconds.
# Workers silent for more than this many seconds are killed and restarted.
# Can be overridden by the GUNICORN_TIMEOUT environment variable.
timeout = int(os.environ.get('GUNICORN_TIMEOUT', '120'))

# Timeout for graceful workers restart.
# After receiving a restart signal, workers have this much time to finish serving requests.
# Can be overridden by the GUNICORN_GRACEFUL_TIMEOUT environment variable.
graceful_timeout = int(os.environ.get('GUNICORN_GRACEFUL_TIMEOUT', '120'))

# Example for Uvicorn specific settings if needed (though often handled by worker_class)
# These settings can be passed directly to the Uvicorn worker.
# uvicorn_kwargs = {
# "loop": "auto", # The event loop implementation ("auto", "asyncio", "uvloop")
# "http": "auto", # The HTTP protocol implementation ("auto", "h11", "httptools")
# }
# Note: Uvicorn-specific settings like 'loop' and 'http' are generally configured
# by the UvicornWorker itself or its defaults. If specific Uvicorn features are needed,
# they can be explored here, but for most FastAPI applications, the defaults are suitable.
