# Trading Platform Installation Guide (Single VPS Production Setup)

This guide provides step-by-step instructions for deploying the Trading Platform application to a single Virtual Private Server (VPS) for a production environment. It assumes you are using Docker and Docker Compose for the backend, and Nginx to serve the frontend and proxy API requests.

## 1. Prerequisites

*   **VPS:** A Linux VPS (e.g., Ubuntu 20.04/22.04) with root or sudo access.
*   **Domain Name:** A registered domain name (e.g., `yourdomain.com`) pointed to your VPS's IP address.
*   **Docker:** Docker installed on the VPS. Follow the official Docker installation guide for your Linux distribution.
*   **Docker Compose:** Docker Compose installed on the VPS. Follow the official Docker Compose installation guide.
*   **Git:** Git installed on the VPS (`sudo apt update && sudo apt install git`).
*   **Basic Linux Command Line Knowledge.**

## 2. Clone the Repository

Clone the project repository to your VPS:
```bash
git clone <your_repository_url>
cd <repository_directory_name>
```

## 3. Backend Setup (Docker Compose)

The backend (FastAPI application, Celery workers, PostgreSQL database, Redis) will run using Docker Compose.

### 3.1. Configure Environment Variables

All backend configurations are managed via environment variables. The production Docker Compose file (`docker-compose.prod.yml`) expects these to be present in the environment where `docker-compose` is run, or in a `.env` file located at `backend/.env`.

1.  **Navigate to the backend directory:**
    ```bash
    cd backend
    ```
2.  **Create a `.env` file from the example:**
    ```bash
    cp .env.example .env
    ```
3.  **Edit the `.env` file (`nano .env` or `vim .env`):**
    Update the following variables with your **production values**. **Do NOT use default or example values for sensitive information.**

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

    **Important Note on Secrets:** For enhanced security in production, consider injecting these secrets directly as environment variables at runtime via your VPS provider's interface or a secrets management tool, rather than solely relying on the `.env` file. However, for a single VPS setup, a well-protected `.env` file (with appropriate file permissions `chmod 600 backend/.env`) is a common starting point.

4.  **Return to the project root directory:**
    ```bash
    cd ..
    ```

### 3.2. Build and Run Backend Services

1.  **Pull the latest images and build your application images:**
    ```bash
    docker-compose -f docker-compose.prod.yml pull
    docker-compose -f docker-compose.prod.yml build
    ```
2.  **Start all backend services in detached mode:**
    ```bash
    docker-compose -f docker-compose.prod.yml up -d
    ```
    This will start the `db` (PostgreSQL), `redis`, `web` (FastAPI/Gunicorn), and `worker` (Celery) services.

### 3.3. Run Database Migrations

After the `db` service is running, apply database migrations using Alembic. The `web` service in `docker-compose.prod.yml` is configured to run migrations on startup (`alembic upgrade head`).

To verify or manually run migrations (if needed):
1.  **Check current migration status (optional):**
    ```bash
    docker-compose -f docker-compose.prod.yml exec web alembic current
    ```
2.  **Apply migrations (if not automatically done or if you need to ensure it):**
    ```bash
    docker-compose -f docker-compose.prod.yml exec web alembic upgrade head
    ```
    If you made changes to `backend/models.py` (like adding `payment_options_json`), ensure you have generated a new migration revision locally (`alembic revision -m "your_migration_message"`) and committed it to your repository before deploying. The `upgrade head` command will then apply it.

## 4. Frontend Setup (Nginx)

Nginx will serve the static frontend files and act as a reverse proxy for the backend API.

### 4.1. Prepare Frontend Files

1.  The frontend files are in the `frontend/` directory of your repository.
2.  Ensure `frontend/js/config.js` has `window.BACKEND_API_BASE_URL = "";`. This is crucial for Nginx to proxy API requests correctly under the same domain.
3.  Create a directory on your VPS to host these files:
    ```bash
    sudo mkdir -p /var/www/yourdomain.com/html
    ```
4.  Copy the contents of the `frontend/` directory to this new location. You can use `rsync`, `scp`, or copy them directly if you cloned the repo into `/var/www/yourdomain.com/html`'s parent.
    Example using `rsync` (run from your project root on the VPS):
    ```bash
    sudo rsync -av frontend/ /var/www/yourdomain.com/html/
    ```
    Ensure correct permissions for Nginx to read these files.
    ```bash
    sudo chown -R www-data:www-data /var/www/yourdomain.com/html
    sudo find /var/www/yourdomain.com/html -type d -exec chmod 755 {} \;
    sudo find /var/www/yourdomain.com/html -type f -exec chmod 644 {} \;
    ```

### 4.2. Configure Nginx

1.  **Use the Nginx Configuration Example:**
    The repository includes `nginx.conf.example`. Copy this to your Nginx sites-available directory, renaming it for your domain:
    ```bash
    sudo cp nginx.conf.example /etc/nginx/sites-available/yourdomain.com.conf
    ```
2.  **Edit the Nginx Configuration:**
    ```bash
    sudo nano /etc/nginx/sites-available/yourdomain.com.conf
    ```
    *   Replace all instances of `yourdomain.com` with your actual domain name.
    *   Replace `www.yourdomain.com` with your actual `www` subdomain or remove it if not used.
    *   Update the `root` directive if you chose a different path for your frontend files (e.g., `root /var/www/yourdomain.com/html;`).
    *   The SSL certificate paths are placeholders (`/etc/letsencrypt/live/yourdomain.com/...`). You will set these up in the next step.

3.  **Enable the Site Configuration:**
    Create a symbolic link from `sites-available` to `sites-enabled`:
    ```bash
    sudo ln -s /etc/nginx/sites-available/yourdomain.com.conf /etc/nginx/sites-enabled/
    ```
    Remove the default Nginx configuration if it exists:
    ```bash
    sudo rm /etc/nginx/sites-enabled/default # If it exists
    ```

### 4.3. Set Up SSL with Certbot (Let's Encrypt)

1.  **Install Certbot and the Nginx plugin:**
    ```bash
    sudo apt update
    sudo apt install certbot python3-certbot-nginx
    ```
2.  **Obtain and Install SSL Certificate:**
    Run Certbot, which will automatically detect your domain from the Nginx configuration and configure SSL.
    ```bash
    sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com # Include -d for each domain/subdomain
    ```
    Follow the prompts. Certbot will modify your Nginx configuration to include the SSL certificate paths and set up automatic renewal.

3.  **Test Nginx Configuration and Reload:**
    After Certbot finishes, test your Nginx configuration:
    ```bash
    sudo nginx -t
    ```
    If the test is successful, reload Nginx to apply the changes:
    ```bash
    sudo systemctl reload nginx
    ```

## 5. Final Checks and Basic Troubleshooting

### 5.1. Verify Services

1.  **Check Docker Containers:**
    ```bash
    docker-compose -f docker-compose.prod.yml ps
    ```
    All services (`db`, `redis`, `web`, `worker`) should show `Up` or `running`.
2.  **Access Your Application:**
    Open your domain (`https://yourdomain.com`) in a web browser. You should see the frontend, and API calls should work.
3.  **Check Nginx Status:**
    ```bash
    sudo systemctl status nginx
    ```
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
5.  **Check Nginx Logs:**
    *   Access log: `/var/log/nginx/yourdomain.com.access.log`
    *   Error log: `/var/log/nginx/yourdomain.com.error.log`
    (These paths are defined in your `yourdomain.com.conf` Nginx file).

### 5.2. Firewall

Ensure your VPS firewall (e.g., `ufw`) allows traffic on ports 80 (HTTP) and 443 (HTTPS).
```bash
sudo ufw allow 'Nginx Full' # Or sudo ufw allow 80; sudo ufw allow 443;
sudo ufw enable # If not already enabled
sudo ufw status
```

## 6. Keeping the Application Running (Daemonizing)

Docker Compose in detached mode (`-d`) already runs the containers in the background. If your VPS reboots, you might need to ensure Docker services start automatically. Most modern Linux distributions configure Docker to start on boot.

To manually start after a reboot if needed:
```bash
cd /path/to/your/repository_directory_name
docker-compose -f docker-compose.prod.yml up -d
```

## 7. Updates and Maintenance

*   **Updating the Application:**
    1.  Navigate to your project directory.
    2.  `git pull` to get the latest code.
    3.  Rebuild Docker images if backend code changed: `docker-compose -f docker-compose.prod.yml build`
    4.  Restart services: `docker-compose -f docker-compose.prod.yml up -d --remove-orphans`
    5.  Run new database migrations if any: `docker-compose -f docker-compose.prod.yml exec web alembic upgrade head`
    6.  If frontend files changed, re-sync them to `/var/www/yourdomain.com/html/`.
*   **Security Updates:** Regularly update your VPS packages (`sudo apt update && sudo apt upgrade`), Docker, and Docker Compose.

This guide provides a foundational setup. For long-term production use, refer to the `PRODUCTION_CHECKLIST.MD` for further enhancements regarding security, monitoring, backups, and scalability.
