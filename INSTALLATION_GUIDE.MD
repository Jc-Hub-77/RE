# Trading Platform Installation Guide (Production Setup with Docker)

This guide provides step-by-step instructions for deploying the Trading Platform application to a production environment using Docker and Docker Compose. This method is recommended for consistency and ease of management.

**Deployment Options Covered:**
1.  **Linux VPS (Recommended):** The primary focus of this guide, detailing deployment on a standard Linux-based Virtual Private Server.
2.  **Windows VPS with Docker Desktop:** Adaptations for deploying on a Windows Server using Docker Desktop with WSL 2.

**Preamble for Windows Docker Users:**

If you are deploying on a Windows Server (e.g., Windows Server 2019/2022), this guide can be adapted by using **Docker Desktop for Windows** with the **WSL 2 (Windows Subsystem for Linux 2) backend**.
*   **Docker Desktop on Windows:** Manages your Docker engine and provides an interface for Docker operations.
*   **WSL 2:** Allows Docker Desktop to run Linux containers efficiently on Windows. It's highly recommended over the older Hyper-V backend for performance and compatibility.
*   Most Docker and Docker Compose commands in this guide will be run from **PowerShell or Windows Terminal** on your Windows host, interacting with Docker Desktop.
*   File paths for volume mounts in `docker-compose.yml` files might need adjustment if you are not placing the project directly within the WSL 2 filesystem accessible to Docker. However, Docker Desktop generally handles path conversions well.

## 1. Prerequisites

### For Linux VPS (Recommended Production Environment):
*   **VPS:** A Linux VPS (e.g., Ubuntu 20.04/22.04) with root or sudo access.
*   **Domain Name:** A registered domain name (e.g., `yourdomain.com`) pointed to your VPS's IP address.
*   **Docker:** Docker Engine installed on the VPS. Follow the official Docker installation guide for your Linux distribution.
*   **Docker Compose:** Docker Compose (v2 or later) installed on the VPS. Follow the official Docker Compose installation guide.
*   **Git:** Git installed on the VPS (`sudo apt update && sudo apt install git`).
*   **Basic Linux Command Line Knowledge.**

### For Windows VPS with Docker Desktop:
*   **Operating System:** Windows Server 2019/2022 (or Windows 10/11 Pro/Enterprise with Hyper-V/WSL 2 support for testing).
*   **Docker Desktop:** Install Docker Desktop for Windows, configured to use the WSL 2 backend (recommended) or Hyper-V. Ensure WSL 2 is installed and enabled.
*   **Git for Windows:** Install Git for Windows ([git-scm.com](https://git-scm.com/download/win)).
*   **Terminal:** Windows Terminal or PowerShell for running commands.
*   **Domain Name:** A registered domain name (e.g., `yourdomain.com`) pointed to your VPS's public IP address.
*   **Administrative Privileges:** For installing software and configuring Docker Desktop.

## 2. Clone the Repository

Clone the project repository to your server:

*   **Linux:**
    ```bash
    git clone <your_repository_url>
    cd <repository_directory_name>
    ```
*   **Windows (using Git Bash, PowerShell, or CMD):**
    *   Choose a suitable directory, e.g., `C:\Projects\`.
    *   ```shell
        git clone <your_repository_url> trading_platform
        cd trading_platform
        ```

## 3. Backend Setup (Docker Compose)

The backend (FastAPI application, Celery workers, PostgreSQL database, Redis) will run using Docker Compose. The commands are generally the same whether on Linux or Windows (using PowerShell/CMD with Docker Desktop).

### 3.1. Configure Environment Variables

All backend configurations are managed via environment variables. For **production environments, these variables should be injected directly by your deployment platform or server environment.** This is the most secure method. The production Docker Compose file (`docker-compose.prod.yml`) is designed to use these externally provided variables. Refer to `SECRETS.MD` for more details on secret management strategies.

**Using `backend/.env` for Local Development or Simpler Windows Setups:**

For local development or simpler single Windows VPS setups where direct environment injection for Docker is complex, you can use a `.env` file located at `backend/.env` on your **Windows host machine**.
1.  Navigate to the `backend` directory (e.g., `C:\Projects\trading_platform\backend` on Windows, or `/path/to/project/backend` on Linux).
2.  Create a `.env` file from the example:
    *   Linux: `cp .env.example .env`
    *   Windows (PowerShell): `Copy-Item .env.example .env`
3.  Edit the `.env` file (e.g., with `nano`, `vim`, or Notepad/VS Code on Windows) with your actual production or development settings.
    *   The `docker-compose.prod.yml` file might be configured to use an `env_file` directive pointing to this `.env` file, or it might expect variables to be passed through the shell environment if `env_file` is commented out or not present. **Check your `docker-compose.prod.yml`.**
    *   If `docker-compose.prod.yml` **does not** use `env_file: backend/.env`, then these variables must be set in the shell environment where you run `docker-compose` commands, or via other Docker environment variable injection methods.

**Using `backend/.env` in Production (Discouraged for Linux, a common pattern for Docker Desktop on Windows if not using advanced orchestration):**

If using a `backend/.env` file directly with Docker Compose in production (especially on Windows where it's simpler than system-wide env vars for Docker):
*   **Secure the file:** On Linux, `chmod 600 backend/.env`. On Windows, ensure appropriate file permissions via Properties -> Security.
*   **Exclude from Docker image:** Ensure your `backend/.dockerignore` (or global `.gitignore` if not building images directly on the VPS) prevents the `.env` file from being copied into any Docker image if you build images on the host. The provided `.dockerignore` should already do this.
*   **Acknowledge the risk:** While convenient, it's less secure than injecting environment variables directly into the container runtime without a file on disk.

**Variables to be Set:**

> **IMPORTANT: Secure Your Production Configuration!**
>
> The environment variables listed below include placeholders and examples (e.g., in `backend/.env.example` or default values in `backend/config.py`). **These are NOT secure for production and MUST be replaced with your own unique, strong, and secret values.**
>
> Failure to do so will expose your application to significant security risks.
>
> **Key variables requiring your immediate attention for production:**
> *   `JWT_SECRET_KEY`: Default example values are insecure. Generate a new one.
> *   `API_ENCRYPTION_KEY`: Default example values are insecure. Generate a new one.
> *   `POSTGRES_PASSWORD`: Use a strong, unique password.
> *   All SMTP credentials (`SMTP_USER`, `SMTP_PASSWORD`).
> *   All Payment Gateway credentials (e.g., `COINBASE_COMMERCE_API_KEY`, `COINBASE_COMMERCE_WEBHOOK_SECRET`).
> *   `ALLOWED_ORIGINS`: Set this to your specific frontend domain(s).
> *   `FRONTEND_URL`: Set this to your specific frontend URL.
>
> Review **every** variable to ensure it's correctly configured for your production environment. Refer to `backend/SECRETS.MD` for more details on secrets management best practices.

Regardless of the method chosen for setting them (preferably direct injection for production), the following variables need to be configured with your **production values**. **Do NOT use default or example values for sensitive information.**

    **Critical Variables:**
    *   `POSTGRES_USER`: Your desired PostgreSQL username.
    *   `POSTGRES_PASSWORD`: A strong, unique password for the PostgreSQL user.
    *   `POSTGRES_DB`: Your desired PostgreSQL database name.
    *   `JWT_SECRET_KEY`: A long, random, and unique string for JWT token generation. Generate one using `openssl rand -hex 32`.
    *   `API_ENCRYPTION_KEY`: A long, random, and unique string for encrypting sensitive data like API keys. Generate one using `openssl rand -hex 32`.
    *   `ALLOWED_ORIGINS`: Your domain name(s) for CORS, comma-separated. Example: `https://yourdomain.com,https://www.yourdomain.com`
    *   `FRONTEND_URL`: The primary URL of your frontend. Example: `https://yourdomain.com`

    **Email (SMTP) Settings (for notifications, password resets, etc.):**
    *   `SMTP_TLS`: `true` or `false`
    *   `SMTP_PORT`: e.g., `587`
    *   `SMTP_HOST`: Your SMTP server hostname (e.g., `smtp.mailgun.org`).
    *   `SMTP_USER`: Your SMTP username.
    *   `SMTP_PASSWORD`: Your SMTP password.
    *   `EMAILS_FROM_EMAIL`: The "From" email address (e.g., `noreply@yourdomain.com`).
    *   `EMAILS_FROM_NAME`: The "From" name (e.g., `Trading Platform`).

    **Payment Gateway (Coinbase Commerce - if used):**
    *   `COINBASE_COMMERCE_API_KEY`: Your Coinbase Commerce API key.
    *   `COINBASE_COMMERCE_WEBHOOK_SECRET`: Your Coinbase Commerce webhook shared secret.
    *   `APP_PAYMENT_SUCCESS_URL`: e.g., `https://yourdomain.com/payment/success`
    *   `APP_PAYMENT_CANCEL_URL`: e.g., `https://yourdomain.com/payment/cancel`

    **Other Settings:**
    *   `STRATEGIES_DIR`: Should typically remain `/app/strategies` as per Docker setup, unless you have custom volume mounts for strategies.
    *   `GUNICORN_WORKERS`: (Optional, defaults to 4) Number of Gunicorn workers. A good starting point is `(2 * CPU_CORES) + 1`.
    *   `CELERY_CONCURRENCY`: (Optional, defaults to 8) Number of concurrent tasks per Celery worker container.
    *   `CELERY_WORKER_REPLICAS`: (Optional, set in `docker-compose.prod.yml` if not overridden by environment, defaults to 2) Number of Celery worker containers.
    *   `ENVIRONMENT`: Set to `production`.

    **After configuring your environment variables (either externally or via `backend/.env`):**

If you created/modified a `.env` file in the `backend` directory, return to the project root (e.g., `C:\Projects\trading_platform` or `/path/to/project`). On Linux:
```bash
cd ..
```
On Windows, ensure you are in the root directory of the cloned project where `docker-compose.prod.yml` is located.

### 3.2. Build and Run Backend Services

These commands are run from the project's root directory (where `docker-compose.prod.yml` is located) in your terminal (PowerShell/CMD on Windows, bash on Linux).

1.  **Pull the latest base images and build your application images:**
    ```bash
    docker-compose -f docker-compose.prod.yml pull
    docker-compose -f docker-compose.prod.yml build
    ```
2.  **Start all backend services in detached mode:**
    ```bash
    docker-compose -f docker-compose.prod.yml up -d
    ```
    This will start the `db` (PostgreSQL), `redis`, `web` (FastAPI/Gunicorn), and `worker` (Celery) services as defined in your `docker-compose.prod.yml`.

### 3.3. Run Database Migrations

After the `db` service (PostgreSQL container) is running, apply database migrations using Alembic. The `web` service in `docker-compose.prod.yml` is typically configured to run migrations on startup (e.g., via an `entrypoint.sh` script that calls `alembic upgrade head`).

To verify or manually run migrations if needed:
1.  **Check current migration status (optional):**
    ```bash
    docker-compose -f docker-compose.prod.yml exec web alembic current
    ```
2.  **Apply migrations (if not automatically done or if you need to ensure it):**
    ```bash
    docker-compose -f docker-compose.prod.yml exec web alembic upgrade head
    ```
    If you made changes to `backend/models.py` (like adding new fields or tables), ensure you have generated a new migration revision locally (`alembic revision -m "your_migration_message"`) and committed it to your repository before deploying. The `upgrade head` command will then apply it.

## 4. Frontend Setup

The frontend consists of static HTML, CSS, and JavaScript files. These can be served by Nginx, either running as a Docker container or directly on the host.

### 4.1. Prepare Frontend Files

1.  The frontend files are in the `frontend/` directory of your repository.
2.  Ensure `frontend/js/config.js` has `window.BACKEND_API_BASE_URL = "";`. This setting is crucial for the reverse proxy setup, as it makes API calls from the frontend use relative paths (e.g., `/api/v1/...`), which Nginx can then correctly route to the backend service.

### 4.2. Configure and Run Web Server (Nginx)

**Option 1: Running Nginx as a Docker Container (Recommended for Dockerized Setup)**

This approach keeps all parts of your application containerized and simplifies networking.

1.  **Create/Update Nginx Configuration:**
    *   Locate `nginx.conf.example` in the project root. Rename it to `nginx.conf` (or create a new `nginx.conf`).
    *   Edit this `nginx.conf` file. The key part is the `proxy_pass` directive. It should point to your backend Docker service name and port as defined in `docker-compose.prod.yml` (usually `web:8000`).
        ```nginx
        # Inside your server block, for the API location:
        location /api/v1/ {
            proxy_pass http://web:8000; # 'web' is the service name in docker-compose
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            # ... other proxy headers as in nginx.conf.example
        }

        # For serving static frontend files:
        location / {
            root /usr/share/nginx/html; # Default root in many Nginx images
            index index.html index.htm;
            try_files $uri $uri/ /index.html; # For SPAs
        }
        ```
2.  **Add Nginx Service to `docker-compose.prod.yml`:**
    Add a new service definition for Nginx to your `docker-compose.prod.yml` file:
    ```yaml
    services:
      # ... your existing db, redis, web, worker services ...

      nginx:
        image: nginx:latest # Or a specific stable version
        container_name: trading_platform_nginx
        ports:
          - "80:80" # Expose HTTP port
          - "443:443" # Expose HTTPS port (for SSL later)
        volumes:
          - ./frontend:/usr/share/nginx/html:ro # Mount your frontend files
          - ./nginx.conf:/etc/nginx/nginx.conf:ro # Mount your Nginx configuration
          # For SSL (covered later):
          # - /path/to/your/ssl_certs_on_host:/etc/letsencrypt:ro
          # - /path/to/your/certbot_www_root_on_host:/var/www/certbot:ro
        depends_on:
          - web # Ensure backend is started before Nginx (though Nginx will retry)
        restart: unless-stopped
    ```
    *   Adjust volume paths as necessary. For Windows, Docker Desktop handles path conversions from Windows paths (e.g., `C:\Projects\trading_platform\frontend`) to Linux paths inside the container.
3.  **Start/Restart Docker Compose:**
    After adding the Nginx service and configuring `nginx.conf`, run:
    ```bash
    docker-compose -f docker-compose.prod.yml up -d --build nginx # To build/rebuild nginx if needed and start
    # Or if all services need restarting:
    # docker-compose -f docker-compose.prod.yml up -d --build
    ```

**Option 2: Running Nginx or IIS Natively on the Windows Host**

This is an alternative if you prefer not to run Nginx inside Docker.

*   **Nginx on Windows Host:**
    1.  Install Nginx on Windows as per Nginx.org documentation (usually involves downloading a zip and running `nginx.exe`).
    2.  Configure your `nginx.conf` file on the Windows host.
        *   The `root` directive for serving frontend files will point to a Windows path (e.g., `C:/Projects/trading_platform/frontend`).
        *   The `proxy_pass` directive for `/api/v1/` will point to `http://localhost:<host_port_for_web_container>`, where `<host_port_for_web_container>` is the port you mapped for the `web` service in `docker-compose.prod.yml` (e.g., if `ports: - "8001:8000"` for `web` service, then proxy to `http://localhost:8001`).
    3.  Run Nginx as a service on Windows (e.g., using NSSM).
*   **IIS on Windows Host:**
    1.  Ensure IIS is installed with Application Request Routing (ARR) and URL Rewrite modules.
    2.  Create a new website in IIS Manager, pointing its physical path to your `frontend` directory.
    3.  Use URL Rewrite to create a reverse proxy rule for the `/api/v1/` path, forwarding requests to `http://localhost:<host_port_for_web_container>`.

**Recommendation:** Option 1 (Nginx in Docker) is generally simpler to manage and configure in a fully Dockerized deployment, as networking between containers (`http://web:8000`) is straightforward.

### 4.3. Set Up SSL with Certbot (Let's Encrypt)

Securing your application with HTTPS is crucial.

*   **If using Nginx in Docker (Option 1 from 4.2):**
    *   You can integrate Certbot directly into your Nginx setup within Docker. This often involves:
        1.  Adding a Certbot service to your `docker-compose.prod.yml` to obtain/renew certificates.
        2.  Sharing a volume between Certbot and Nginx for SSL certificates (e.g., `/etc/letsencrypt`).
        3.  Sharing a volume for Certbot's webroot challenges (e.g., `/var/www/certbot`).
        4.  Modifying your `nginx.conf` to handle ACME challenges and use the SSL certificates.
    *   Alternatively, obtain certificates on the Docker host (if Linux) or Windows host (using Certbot for Windows) and mount them into the Nginx container (as shown in the example `nginx` service volumes). This might be simpler for initial setup.
*   **If using Nginx/IIS on the Windows Host (Option 2 from 4.2):**
    *   **Nginx on Windows:** Use Certbot for Windows ([certbot.eff.org](https://certbot.eff.org/)) to obtain certificates and configure Nginx.
    *   **IIS:** Use IIS Manager's built-in tools to create a certificate signing request (CSR) and import a certificate from a Certificate Authority, or use tools like Certify The Web or `win-acme` for Let's Encrypt automation with IIS.

    Update your Nginx/IIS configuration to listen on port 443, specify certificate paths, and redirect HTTP to HTTPS. (Detailed Certbot/SSL steps are extensive and depend on the chosen method; refer to Certbot's official documentation).

## 5. Final Checks and Basic Troubleshooting

### 5.1. Verify Services

1.  **Check Docker Containers (if using Docker for Nginx/backend):**
    (Run from PowerShell or CMD in your project directory on Windows)
    ```bash
    docker-compose -f docker-compose.prod.yml ps
    ```
    All services (`db`, `redis`, `web`, `worker`, and `nginx` if containerized) should show a status indicating they are running (e.g., `Up`, `running`).
2.  **Access Your Application:**
    Open your domain (`https://yourdomain.com`) in a web browser. You should see the frontend, and API calls should work.
3.  **Check Nginx/IIS Logs:**
    *   **Nginx (Docker):** `docker-compose -f docker-compose.prod.yml logs nginx`
    *   **Nginx (Windows Host):** Check logs in the Nginx `logs` directory (e.g., `C:\nginx\logs`).
    *   **IIS (Windows Host):** Check logs in `%SystemDrive%\inetpub\logs\LogFiles`.
4.  **Check Backend Logs:**
    If you encounter issues with the backend API:
    ```bash
    docker-compose -f docker-compose.prod.yml logs web
    docker-compose -f docker-compose.prod.yml logs worker
    ```
    For continuous logs:
    ```bash
    docker-compose -f docker-compose.prod.yml logs -f web
    ```

### 5.2. Firewall

*   **Linux VPS:** Ensure your VPS firewall (e.g., `ufw`) allows traffic on ports 80 (HTTP) and 443 (HTTPS).
    ```bash
    sudo ufw allow 'Nginx Full' # Or sudo ufw allow 80; sudo ufw allow 443;
    sudo ufw enable # If not already enabled
    sudo ufw status
    ```
*   **Windows VPS (with Docker Desktop):**
    *   Docker Desktop for Windows typically manages host firewall rules for exposed container ports when using WSL 2.
    *   However, ensure Windows Firewall is configured to allow inbound traffic on ports 80 and 443, especially if your Nginx/web server is running directly on the Windows host or if there are specific network configurations.
    *   Check Windows Defender Firewall -> Advanced Settings -> Inbound Rules. Add rules for TCP ports 80 and 443 if necessary.

## 6. Keeping the Application Running

*   **Linux VPS:** Docker Compose in detached mode (`-d`) runs containers in the background. Most Linux distributions configure Docker to start on boot. To ensure your application containers start on boot, set `restart: unless-stopped` or `restart: always` for your services in `docker-compose.prod.yml`.
*   **Windows VPS with Docker Desktop:** Docker Desktop can be configured to start when you log into Windows. For services to start automatically with Docker Desktop, ensure your Docker Compose services have a `restart` policy like `unless-stopped` or `always` in `docker-compose.prod.yml`.

To manually start after a reboot if needed (from your project directory):
```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 7. Updates and Maintenance

*   **Updating the Application:**
    1.  Navigate to your project directory (e.g., `C:\Projects\trading_platform` on Windows).
    2.  `git pull` to get the latest code.
    3.  Rebuild Docker images if backend or frontend code changed: `docker-compose -f docker-compose.prod.yml build`
    4.  Restart services: `docker-compose -f docker-compose.prod.yml up -d --remove-orphans`
    5.  Run new database migrations if any: `docker-compose -f docker-compose.prod.yml exec web alembic upgrade head`
    6.  If frontend files are served by a host Nginx/IIS (not the Docker Nginx), re-sync them to the appropriate web server directory.
*   **Security Updates:**
    *   **Linux:** Regularly update your VPS packages (`sudo apt update && sudo apt upgrade`), Docker, and Docker Compose.
    *   **Windows:** Regularly apply Windows Updates. Keep Docker Desktop and Git for Windows updated.

This guide provides a foundational setup. For long-term production use, refer to the `PRODUCTION_CHECKLIST.MD` for further enhancements regarding security, monitoring, backups, and scalability.
