# Windows Deployment Guide for Trading Platform

## Introduction

The Trading Platform is a comprehensive application designed to facilitate automated trading strategies, backtesting, and user management. This guide provides step-by-step instructions for deploying the platform (backend and frontend) on a Windows Server environment. It specifically focuses on a native Windows deployment for all components where feasible, without relying on Docker or containerization technologies.

The main components you will be installing and configuring are:
- **Backend:** Python-based FastAPI application.
- **Task Queuing:** Celery workers for handling asynchronous tasks (e.g., live trading strategies, backtests).
- **Database:** PostgreSQL for data storage.
- **Message Broker/Cache:** Redis for Celery and potentially caching.
- **Web Server:** Nginx (recommended) or IIS, to serve the frontend static files and act as a reverse proxy for the backend API.

## 1. Prerequisites

Before starting the deployment process, ensure your Windows Server environment meets the following prerequisites:

*   **Operating System:**
    *   Recommended: Windows Server 2019 or Windows Server 2022.
    *   Modern Windows desktop versions (Windows 10 Pro/Enterprise, Windows 11 Pro/Enterprise) may work for testing or small-scale deployments but are not recommended for production environments due to limitations in concurrent connections, resource management, and stability features.

*   **Administrative Privileges:**
    *   You must have full administrative privileges on the server to install software, configure services, modify firewall rules, and manage system settings.

*   **System Resources (Initial Estimates):**
    *   These are initial estimates and can vary significantly based on the number of users, concurrently active trading strategies, data volume (especially historical price data and trade logs), backtesting frequency, and overall user activity.
    *   **CPU:** Multi-core processor (e.g., 4-8+ vCPUs). The FastAPI backend is generally efficient, but Celery workers (especially if running multiple complex strategies simultaneously) and PostgreSQL can be CPU-intensive.
    *   **RAM:** Significant RAM (e.g., 16GB - 32GB or more). Redis, PostgreSQL, Python processes (FastAPI app, multiple Celery workers), and potentially many concurrent trading strategies will consume memory. Insufficient RAM can lead to performance degradation.
    *   **Disk Space:** Fast storage (SSD/NVMe) is highly recommended for optimal performance, especially for the database and logs.
        *   Initial estimate: 100GB - 200GB+ of free space. This should cover the OS, installed software, the application's Python environment, logs, and initial database size.
        *   Database storage can grow significantly depending on the amount of historical data stored and the volume of trades. Plan for monitoring and expansion.
    *   **Network:** A reliable, high-speed, and stable internet connection is crucial. Low latency to the exchanges your users will connect to is highly beneficial for live trading.

    *   **Important Caveat:** The resource estimates above are starting points. For a production environment, especially one targeting a significant number of users or complex strategies, **rigorous load testing and continuous performance monitoring are essential** to determine the actual resource requirements and identify bottlenecks. Be prepared to scale resources as needed.

*   **Software Downloads:**
    *   You will need to download the installers or binaries for:
        *   Python (from python.org)
        *   PostgreSQL (from postgresql.org or EDB installers)
        *   Redis (e.g., the MSOpenTech archive for native Windows, or consider Redis on WSL for more up-to-date versions if native stability is a concern)
        *   Nginx (from nginx.org) or ensure IIS is enabled.
        *   NSSM (Non-Sucking Service Manager, from nssm.cc) or an alternative process manager.
        *   Git (from git-scm.com)

*   **Domain Name:**
    *   A registered domain name (e.g., `yourtradingplatform.com`) is required for public access.
    *   DNS records for this domain must be configured to point to your Windows VPS's public IP address. This is essential for SSL certificate setup and user accessibility.

*   **Firewall Access:**
    *   You need the ability to configure the Windows Defender Firewall (or any other firewall in use) to allow inbound traffic on specific ports (typically HTTP/80 and HTTPS/443).

*   **Basic Windows Server Administration Knowledge:**
    *   Familiarity with managing Windows services, editing environment variables, navigating the file system, using PowerShell or Command Prompt, and potentially configuring IIS (if chosen as the web server) will be very helpful.

## 2. Software Installation

This section covers the installation of all required third-party software components on your Windows Server. It's recommended to download installers from official sources to ensure their integrity.

### 2.1. Python

The backend application requires Python. We recommend using a recent stable version such as Python 3.10.x or 3.11.x. (Check the project's `requirements.txt` or documentation for any specific version compatibility notes, though the application aims for compatibility with recent Python 3 versions).

*   **Download:**
    *   Go to the [official Python downloads page for Windows](https://www.python.org/downloads/windows/).
    *   Download the "Windows installer (64-bit)" for the recommended Python version.

*   **Installation Steps:**
    1.  Run the downloaded installer as an Administrator.
    2.  **Crucial:** On the first screen of the installer, ensure you check the box "Add Python X.Y to PATH". This will make `python` and `pip` accessible from the command line.
    3.  Choose "Customize installation" if you want to change the installation location or select optional features. Ensure "pip" is selected for installation. "Install for all users" is generally recommended for a server environment if appropriate for your security policies.
    4.  Proceed through the installation.

*   **Verification:**
    *   Open Command Prompt or PowerShell and type the following commands:
        ```shell
        python --version
        pip --version
        ```
    *   These should output the installed Python and pip versions, respectively.

*   **Virtual Environments:**
    *   While Python is installed system-wide, the application itself will run in a Python virtual environment for better dependency management. This will be covered in the "3. Backend Setup" section.

### 2.2. PostgreSQL

PostgreSQL is the recommended relational database for this platform. We suggest using a recent stable version (e.g., PostgreSQL 14.x, 15.x, or newer).

*   **Download:**
    *   The most straightforward way to install PostgreSQL on Windows is by using the installers provided by EDB (EnterpriseDB). Download them from the [PostgreSQL downloads page (Windows installers)](https://www.postgresql.org/download/windows/).

*   **Installation Steps:**
    1.  Run the EDB PostgreSQL installer as an Administrator.
    2.  When prompted, select the components to install. Typically, you'll need:
        *   `PostgreSQL Server`
        *   `pgAdmin 4` (useful GUI for database management)
        *   `Command Line Tools` (includes `psql`)
    3.  Specify the installation directory and the data directory location (or leave defaults).
    4.  **Set a strong password for the `postgres` superuser.** Remember this password; you'll need it for initial database setup.
    5.  Leave the port as the default (`5432`) unless there's a conflict.
    6.  Select your desired locale or leave it as default.
    7.  Complete the installation.

*   **Post-Installation Configuration:**
    1.  **Accessing `psql` or `pgAdmin`:**
        *   `pgAdmin 4` should be available from the Start Menu. You can use it to connect to your local PostgreSQL server (usually named "PostgreSQL <version>").
        *   To use `psql`, you'll first need to add the PostgreSQL `bin` directory to your system's PATH environment variable (see "Adding PostgreSQL to PATH" below).
    2.  **Creating a Database User:**
        *   Open `psql` (either via pgAdmin's PSQL tool or Command Prompt after setting PATH). You may need to log in as the `postgres` user initially.
        *   Execute the following SQL command to create a dedicated user for the application. Replace `'your_strong_password'` with a strong, unique password:
            ```sql
            CREATE USER trading_platform_user WITH PASSWORD 'your_strong_password';
            ```
    3.  **Creating the Database:**
        *   Execute the following SQL command to create the database, owned by the new user:
            ```sql
            CREATE DATABASE trading_platform_db OWNER trading_platform_user;
            ```
    4.  **Granting Privileges (Optional but Recommended):**
        *   While the owner typically has all privileges, you can explicitly grant them:
            ```sql
            GRANT ALL PRIVILEGES ON DATABASE trading_platform_db TO trading_platform_user;
            ```
    5.  **Configuring `pg_hba.conf` (Client Authentication):**
        *   For the application running on the same server (localhost) to connect to the database, the default settings in `pg_hba.conf` are often sufficient. This file controls which hosts are allowed to connect and which authentication methods are used.
        *   Find `pg_hba.conf` in your PostgreSQL data directory (e.g., `C:\Program Files\PostgreSQL\<version>\data\pg_hba.conf`).
        *   Ensure there's a line allowing `localhost` (or `127.0.0.1`) connections for your new user, typically using `md5` or `scram-sha-256` for password authentication. A default line might look like:
            ```
            # TYPE  DATABASE        USER            ADDRESS                 METHOD
            host    all             all             127.0.0.1/32            scram-sha-256
            host    all             all             ::1/128                 scram-sha-256
            ```
        *   If you change `pg_hba.conf`, you'll need to reload the PostgreSQL server configuration (usually by restarting the PostgreSQL service or using `pg_ctl reload`).
        *   This guide assumes local access. If remote database access were required (not recommended for this single-server setup), you would also need to configure `listen_addresses` in `postgresql.conf`.

*   **Adding PostgreSQL to PATH:**
    1.  Search for "environment variables" in the Windows Start Menu and select "Edit the system environment variables".
    2.  In the System Properties window, click the "Environment Variables..." button.
    3.  Under "System variables", find and select the `Path` variable, then click "Edit...".
    4.  Click "New" and add the path to your PostgreSQL `bin` directory (e.g., `C:\Program Files\PostgreSQL\<YourVersion>\bin`) and the `lib` directory (e.g., `C:\Program Files\PostgreSQL\<YourVersion>\lib` if needed for certain client libraries, though usually `bin` is sufficient for `psql`).
    5.  Click OK on all windows to save the changes. You may need to restart Command Prompt or PowerShell for the changes to take effect.

*   **Verification:**
    *   Open a new Command Prompt or PowerShell and type:
        ```shell
        psql -U trading_platform_user -d trading_platform_db -h localhost
        ```
    *   When prompted for the password, enter the password you set for `trading_platform_user`.
    *   If successful, you should be connected to the `trading_platform_db` database. Type `\q` to exit.

### 2.3. Redis

Redis is used as the message broker for Celery task queuing and can also be utilized for caching.

*   **Recommendation & Download:**
    *   **Native Windows Port (MSOpenTech):** For a direct Windows installation, the MSOpenTech archive of Redis (typically version 3.x) is a common choice. You can find releases on their [GitHub releases page](https://github.com/microsoftarchive/redis/releases). Look for the latest `.msi` or `.zip` file.
    *   **Caveat:** These native Windows ports by MSOpenTech are older versions of Redis and are not officially supported by Redis Labs for production use. They may lack recent features and security patches.
    *   **Alternative (WSL2):** For a more up-to-date and robust Redis experience on Windows, consider installing Redis on Windows Subsystem for Linux (WSL) 2. This allows you to run the latest official Linux version of Redis. Setting up WSL2 is beyond the scope of this guide, but if you are comfortable with it, it's a viable alternative.
    *   This guide will proceed with instructions for the MSOpenTech native Windows version for simplicity in a fully native Windows setup.

*   **Installation (MSOpenTech version):**
    *   **Using the `.msi` installer:**
        1.  Download the `.msi` file (e.g., `Redis-x64-3.0.504.msi`).
        2.  Run the installer as Administrator.
        3.  Follow the prompts. You can typically accept the default installation path (e.g., `C:\Program Files\Redis`).
        4.  Ensure the option to "Add the Redis installation folder to the PATH environment variable" is checked.
        5.  The installer usually sets up Redis as a Windows service automatically.
    *   **Using the `.zip` archive:**
        1.  Download the `.zip` file.
        2.  Extract the contents to a chosen directory (e.g., `C:\Redis`).
        3.  Manually add this directory (e.g., `C:\Redis`) to your system's PATH environment variable (similar steps as adding PostgreSQL to PATH).

*   **Configuration (Basic):**
    *   The configuration file is typically named `redis.windows.conf` (for MSOpenTech MSI install) or `redis.conf` (if using the ZIP or other versions). It's located in the Redis installation directory.
    *   For local use by the application on the same server, the default settings are often sufficient:
        *   `bind 127.0.0.1`: This is the default and crucial for security, ensuring Redis only listens for connections from the local machine.
        *   `port 6379`: The standard Redis port.
    *   **Security Note:** If there's any chance your Redis port might be exposed to external networks (not recommended for this setup), or for enhanced security, set a password using the `requirepass` directive in the configuration file. You would then configure this password in your Celery settings (`CELERY_BROKER_URL` and `CELERY_RESULT_BACKEND_URL`). For a single VPS where Redis is bound to `127.0.0.1`, this is less critical but still good practice.
        ```
        # Example in redis.windows.conf:
        requirepass your_strong_redis_password
        ```
        If you set a password, remember to restart the Redis service.

*   **Running Redis:**
    *   **If installed as a service (MSI default):**
        1.  Open `services.msc` (Windows Services).
        2.  Look for the "Redis" service.
        3.  Ensure it's running and set to "Automatic" startup type. You can start/stop/restart it from here.
    *   **If running manually (from ZIP):**
        1.  Open Command Prompt or PowerShell in your Redis directory (e.g., `C:\Redis`).
        2.  Run: `redis-server.exe redis.windows.conf` (or `redis.conf`).
        3.  This will start Redis in the foreground. You'll need to keep this window open. For production, running it as a service using NSSM (see section 2.6) is highly recommended.

*   **Verification:**
    1.  Open a new Command Prompt or PowerShell (ensure Redis directory is in PATH if not using the service).
    2.  Type `redis-cli.exe`.
    3.  If Redis is running, the prompt will change to `127.0.0.1:6379>`.
    4.  Type `ping`. It should return `PONG`.
    5.  Type `exit` to close `redis-cli`.

### 2.4. Git

Git is required to clone the application repository from its source control.

*   **Download:**
    *   Go to the [official Git for Windows website](https://git-scm.com/download/win).
    *   The download should start automatically, or choose the appropriate installer (usually 64-bit).

*   **Installation Steps:**
    1.  Run the downloaded Git installer as an Administrator.
    2.  Follow the prompts. Key options to consider:
        *   **Select Components:** Defaults are usually fine. "Git Bash Here" and "Git GUI Here" in context menus can be useful.
        *   **Adjusting your PATH environment:** The recommended option is usually "Git from the command line and also from 3rd-party software". This makes Git accessible from Command Prompt, PowerShell, and other tools.
        *   **Choosing HTTPS transport backend:** "Use the OpenSSL library" is a common and good choice.
        *   **Configuring line ending conversions:** The default "Checkout Windows-style, commit Unix-style line endings" is generally recommended for cross-platform projects.
        *   **Configuring the terminal emulator to use with Git Bash:** "Use MinTTY (the default terminal of MSYS2)" is fine.
        *   Other options can usually be left at their defaults unless you have specific preferences.

*   **Verification:**
    *   Open a new Command Prompt or PowerShell.
    *   Type:
        ```shell
        git --version
        ```
    *   This should output the installed Git version.

### 2.5. Web Server / Reverse Proxy (Nginx or IIS)

A web server is essential for serving the static frontend files (HTML, CSS, JavaScript) and acting as a reverse proxy for the backend FastAPI application. This setup improves security, allows for easier SSL termination, and can handle load balancing or caching if needed in the future. This guide provides instructions for Nginx and briefly for IIS.

**Option 1: Nginx on Windows (Recommended)**

Nginx is a lightweight, high-performance web server and reverse proxy.

*   **Download:**
    *   Go to the [Nginx for Windows download page](http://nginx.org/en/download.html).
    *   Download the latest stable version (e.g., "Mainline version"). It will be a `.zip` file.

*   **Installation:**
    1.  Create a directory for Nginx (e.g., `C:\nginx`).
    2.  Extract the contents of the downloaded Nginx `.zip` file into this directory. You should see subdirectories like `conf`, `html`, `logs`, etc.
    3.  Nginx is run directly from this folder; there's no separate installer.

*   **Basic Nginx Commands:**
    *   Open Command Prompt or PowerShell in the Nginx directory (e.g., `cd C:\nginx`).
    *   Start Nginx: `start nginx.exe` or simply `nginx.exe` (this will run it in the foreground or background depending on the shell).
    *   Stop Nginx (graceful shutdown): `nginx.exe -s stop`
    *   Reload configuration: `nginx.exe -s reload`
    *   Test configuration: `nginx.exe -t` (very important before reloading a new config).
    *   Quit Nginx (immediate shutdown): `nginx.exe -s quit`

*   **Configuration (Basic Reverse Proxy):**
    1.  The main Nginx configuration file is `conf\nginx.conf` within your Nginx directory (e.g., `C:\nginx\conf\nginx.conf`).
    2.  You will adapt a configuration similar to the `nginx.conf.example` provided in the root of the Trading Platform project repository.
    3.  A typical server block for this application might look like this (this is a simplified example; refer to the project's example for more details):

        ```nginx
        server {
            listen 80; # Or 443 for HTTPS, covered later
            server_name yourtradingplatform.com www.yourtradingplatform.com; # Replace with your domain

            # Serve Frontend Static Files
            location / {
                root C:/path/to/your/frontend/files; # Replace with the actual path to your frontend build/static files
                index index.html index.htm;
                try_files $uri $uri/ /index.html; # Important for single-page applications (SPAs)
            }

            # Reverse Proxy for Backend API
            location /api/v1/ {
                proxy_pass http://127.0.0.1:8000; # Assuming FastAPI runs on port 8000
                proxy_set_header Host $host;
                proxy_set_header X-Real-IP $remote_addr;
                proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
                proxy_set_header X-Forwarded-Proto $scheme;
                proxy_http_version 1.1;
                proxy_set_header Upgrade $http_upgrade;
                proxy_set_header Connection "upgrade";
                # Add other necessary headers like X-CSRF-Token if your app uses them
            }

            # Access and error logs (customize paths)
            access_log logs/platform.access.log;
            error_log logs/platform.error.log;
        }
        ```
    4.  Replace `yourtradingplatform.com`, `C:/path/to/your/frontend/files`, and `http://127.0.0.1:8000` with your actual domain, frontend path, and backend address/port.
    5.  Save the configuration. Always test it with `nginx.exe -t` before applying with `nginx.exe -s reload`.

*   **Running Nginx as a Service:**
    *   Nginx for Windows does not include functionality to run as a Windows service out-of-the-box.
    *   It is **highly recommended** to use a process manager like NSSM (covered in section 2.6) to install and manage Nginx as a Windows service for production. This ensures it starts automatically on server boot and can be restarted if it fails.

**Option 2: IIS (Internet Information Services)**

IIS is Microsoft's web server, built into Windows Server.

*   **Installation:**
    1.  Open **Server Manager**.
    2.  Click "Add roles and features".
    3.  Select "Role-based or feature-based installation".
    4.  Select your server.
    5.  Under "Server Roles", check "Web Server (IIS)".
    6.  Ensure the following features/role services are selected under "Web Server (IIS)":
        *   **Common HTTP Features:** Default Document, Static Content (at a minimum).
        *   **Application Development:** While not directly running ASP.NET, some features pulled in by selecting ASP.NET might be useful for management or if other apps run on IIS. For a pure reverse proxy, this might not be strictly necessary if ARR handles everything.
        *   **Management Tools:** IIS Management Console.
    7.  **Crucial for Reverse Proxy:** You will need **Application Request Routing (ARR)** and **URL Rewrite** modules.
        *   These might not be installed by default with IIS.
        *   You can download and install them using the **Microsoft Web Platform Installer (WebPI)** if available, or find direct downloads from Microsoft's IIS site. Search for "IIS Application Request Routing download" and "IIS URL Rewrite download". Install both.

*   **Configuration (Basic Reverse Proxy):**
    1.  Open **IIS Manager**.
    2.  In the "Connections" pane, expand your server, right-click on "Sites", and select "Add Website...".
        *   **Site name:** e.g., `TradingPlatformFrontend`
        *   **Physical path:** Point this to the directory where your frontend static files are located (e.g., `C:\inetpub\wwwroot\tradingplatform`).
        *   **Binding:** Set the hostname (e.g., `yourtradingplatform.com`) and port (typically 80 for now, SSL later).
    3.  **Enable Proxy Functionality in ARR:**
        *   Select the server node in IIS Manager (top level).
        *   Open "Application Request Routing Cache".
        *   In the "Actions" pane, click "Server Proxy Settings...".
        *   Check "Enable proxy" and click "Apply".
    4.  **Configure URL Rewrite for Reverse Proxy:**
        *   Select your newly created website in IIS Manager.
        *   Open the "URL Rewrite" module.
        *   Click "Add Rule(s)..." in the "Actions" pane.
        *   Choose "Reverse Proxy" and click OK. (If you don't see Reverse Proxy, ensure ARR is installed correctly).
        *   **Inbound Rules:**
            *   Enter the address of your backend server: `127.0.0.1:8000` (or whatever port your FastAPI app runs on).
            *   Click OK. This creates a basic rule.
        *   **Refine the Rule:**
            *   Select the rule and click "Edit...".
            *   Under "Match URL", change the "Pattern" to `^api/v1/(.*)` to only proxy requests starting with `/api/v1/`.
            *   Under "Action", the "Rewrite URL" should be something like `http://127.0.0.1:8000/api/v1/{R:1}`.
            *   Ensure "Stop processing of subsequent rules" is checked for this rule.
        *   You might need another rule (or ensure default behavior) to correctly serve static files from the physical path for other requests.
    5.  **Setting HTTP Headers (ARR):**
        *   When using ARR as a reverse proxy, it generally forwards necessary headers. You can investigate "Server Variables" within ARR settings if specific header manipulation (like `X-Forwarded-Proto` for SSL) is needed, though this is often handled at the SSL termination point.

### 2.6. Process Manager (e.g., NSSM)

A process manager is crucial for running backend applications like the FastAPI server and Celery workers as background services that can automatically restart on failure or server boot. NSSM (Non-Sucking Service Manager) is a popular choice for Windows.

*   **Introduction:**
    *   Python applications (Uvicorn/Gunicorn for FastAPI, Celery workers) are typically console applications. If you run them directly in a Command Prompt or PowerShell window, they will terminate when the window is closed or the user logs out.
    *   A process manager like NSSM allows you to wrap these console applications as proper Windows services.

*   **Download NSSM:**
    *   Go to the [NSSM download page](https://nssm.cc/download).
    *   Download the latest release (it will be a `.zip` file containing 32-bit and 64-bit versions of `nssm.exe`).

*   **Installation (Setup):**
    1.  Create a directory for NSSM (e.g., `C:\NSSM`).
    2.  Extract `nssm.exe` from the downloaded `.zip` file (choose the `win64` version for 64-bit Windows) into the directory you created (e.g., `C:\NSSM\nssm.exe`).
    3.  It's highly recommended to add this directory to your system's PATH environment variable so you can run `nssm` from any command prompt.

*   **Usage (General Steps):**
    1.  Open Command Prompt or PowerShell **as Administrator**.
    2.  To install a new service, use the command:
        ```shell
        nssm install <YourServiceName>
        ```
        (e.g., `nssm install TradingPlatformAPI` or `nssm install CeleryWorkerDefault`)
    3.  This command will open the NSSM GUI configuration window for the new service.
        *   **`Application` Tab:**
            *   **Path:** Full path to the primary executable.
                *   For FastAPI (Uvicorn): This might be the path to `python.exe` in your virtual environment, or directly to `uvicorn.exe` if it's installed globally in the venv's Scripts folder.
                *   For Celery: Path to `python.exe` or `celery.exe` in your virtual environment.
            *   **Startup directory:** The root directory of your backend application (where your `.env` file and `main.py` or Celery app instance are located).
            *   **Arguments:** The command-line arguments needed to run your application.
                *   For Uvicorn: `main:app --host 0.0.0.0 --port 8000` (adjust `main:app` and port as needed).
                *   For Celery Worker: `worker -A backend.celery_app.celery_app -l info -Q default_queue` (adjust queue name and log level).
        *   **`Details` Tab:**
            *   **Display name:** A user-friendly name for the service (e.g., "Trading Platform API Backend").
            *   **Description:** A brief description of the service.
        *   **`I/O` Tab (Optional but Recommended):**
            *   Configure paths for redirecting standard output (stdout) and standard error (stderr) to log files. This is crucial for debugging.
        *   **`Restart` Tab:**
            *   Configure automatic restart options (e.g., restart on application crash, delay between restarts).
        *   Click "Install service".
    4.  **Other Useful NSSM Commands:**
        *   `nssm start <YourServiceName>`
        *   `nssm stop <YourServiceName>`
        *   `nssm restart <YourServiceName>`
        *   `nssm status <YourServiceName>`
        *   `nssm edit <YourServiceName>` (to modify an existing service's configuration)
        *   `nssm remove <YourServiceName>` (to delete a service)

*   Specific instructions for creating services for the FastAPI backend and Celery workers will be provided in the "3. Backend Setup" section.

## 3. Backend Setup

This section details the steps required to set up and configure the backend of the Trading Platform application. Ensure you have completed the software installations from Section 2 before proceeding.

### 3.1. Clone Repository

First, you need to clone the application's source code repository onto your server.

1.  **Choose a Directory:**
    *   Create or choose a suitable directory to house the project files. For example, `C:\Projects\`.
    *   Open Command Prompt or PowerShell and navigate to this directory:
        ```shell
        cd C:\Projects
        ```

2.  **Clone the Repository:**
    *   Use the `git clone` command followed by your repository's URL. Replace `<your_repository_url>` with the actual URL (e.g., from GitHub, GitLab, Bitbucket).
        ```shell
        git clone <your_repository_url> trading_platform
        ```
        This will create a subdirectory named `trading_platform` (or whatever you name it) containing the project files.

3.  **Navigate into Project Directory:**
    ```shell
    cd trading_platform
    ```
    Most subsequent backend commands will be run from within this project directory, specifically from the `backend` subdirectory if your project has a structure like `trading_platform/backend/`. For this guide, we'll assume commands are run from the root of the cloned `trading_platform` directory, and paths will be adjusted accordingly (e.g., `python -m venv backend/venv`). Adjust if your project structure differs.

### 3.2. Python Virtual Environment

Using a Python virtual environment is crucial for isolating the application's dependencies from system-wide Python packages.

1.  **Navigate to Backend Directory (if applicable):**
    *   If your project has a dedicated `backend` folder containing `requirements.txt` and your main Python application, navigate into it:
        ```shell
        cd backend
        ```
    *   If your `requirements.txt` is in the root, you can create the virtual environment there. For clarity, this guide will assume a `backend` subdirectory for Python-specific code.

2.  **Create the Virtual Environment:**
    *   It's good practice to name the virtual environment `venv` or `.venv`.
        ```shell
        python -m venv venv
        ```
        This command creates a `venv` folder within your current directory (e.g., `C:\Projects\trading_platform\backend\venv`).

3.  **Activate the Virtual Environment:**
    *   On Windows, the activation script is located in the `Scripts` subdirectory of your virtual environment.
        ```shell
        .\venv\Scripts\activate
        ```
    *   After activation, your command prompt should be prefixed with `(venv)`, indicating that the virtual environment is active. For example: `(venv) C:\Projects\trading_platform\backend>`.
    *   **Important:** Always ensure your virtual environment is active before installing dependencies or running the application.

### 3.3. Install Dependencies

Once the virtual environment is active, install the Python packages required by the backend.

1.  **Ensure Virtual Environment is Active:** Your command prompt should show `(venv)`.
2.  **Locate `requirements.txt`:** This file lists all necessary Python packages and their versions. It should be in your backend directory (e.g., `C:\Projects\trading_platform\backend\requirements.txt`).
3.  **Install Dependencies:**
    *   Run the following command:
        ```shell
        pip install -r requirements.txt
        ```
    *   This will download and install all packages specified in the `requirements.txt` file into your virtual environment.
    *   **Potential Issues:**
        *   If the installation fails for some packages, it might be due to missing system libraries or compiler tools on Windows for certain Python packages that have C extensions. Check the error messages carefully; you might need to install pre-compiled Windows binaries (`.whl` files) for problematic packages or install Visual C++ Build Tools.
        *   Ensure `requirements.txt` is up-to-date with the latest tested dependencies for the project. (While this guide assumes a given `requirements.txt`, developers should maintain this file using `pip freeze > requirements.txt` after installing or updating packages in their development environment.)

### 3.4. Environment Variables

Environment variables are used to configure the application for the production environment, including database connections, secret keys, and other critical settings. **Never hardcode production secrets directly into the application code.**

*   **Reference Files:**
    *   Refer to the `backend/.env.example` file in your repository for a list of required and optional environment variables.
    *   Consult `backend/SECRETS.md` (if it exists) for guidance on generating secure values for secrets.

*   **How to Set Environment Variables on Windows:**
    *   **Important Note on `.env` files:** While `.env` files are commonly used for local development (and `config.py` is set up to read from one), they should **NOT** be used for production deployment on the server. Instead, set environment variables directly in the system so they are available to the services.
    *   **Method 1: System Properties (GUI) - Recommended for persistent settings:**
        1.  Search for "environment variables" in the Windows Start Menu and select "Edit the system environment variables".
        2.  In the System Properties window, click the "Environment Variables..." button.
        3.  Under "System variables" (recommended for services, as they usually run under system accounts or dedicated service accounts), click "New..." to add a new variable or select an existing one and click "Edit...".
        4.  Enter the `Variable name` (e.g., `DATABASE_URL`) and `Variable value`.
        5.  **Key Variables to Set (examples, refer to your `.env.example`):**
            *   `DATABASE_URL`: `postgresql://trading_platform_user:your_strong_password@localhost:5432/trading_platform_db` (replace with your actual user, password, and DB name).
            *   `JWT_SECRET_KEY`: Generate a long, random, secure string.
            *   `API_ENCRYPTION_KEY`: Generate a strong key for encrypting API keys (e.g., using `Fernet.generate_key().decode()`).
            *   `REDIS_URL`: `redis://localhost:6379/0` (adjust if Redis is password-protected or on a different port/DB). If you set a password in Redis, use: `redis://:your_strong_redis_password@localhost:6379/0`.
            *   `STRATEGIES_DIR`: **Crucial for production.** Set this to an **absolute path** on your Windows server where custom strategy `.py` files will be stored and managed (e.g., `C:\TradingPlatformData\Strategies`). Ensure this directory exists and the application has appropriate permissions to read from it.
            *   `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAILS_FROM_EMAIL`: For email notifications.
            *   `FRONTEND_URL`: The public URL of your frontend (e.g., `https://yourtradingplatform.com`).
            *   `ENVIRONMENT`: Set this to `production`.
            *   Any other API keys or settings required by your application.
        6.  Click OK on all windows to save the changes.

    *   **Method 2: PowerShell (Persistent System-wide):**
        *   Open PowerShell as Administrator.
        *   Use the `SetEnvironmentVariable` command. For system-wide variables available to all users and services:
            ```powershell
            [System.Environment]::SetEnvironmentVariable("VARIABLE_NAME", "your_value", "Machine")
            ```
            Example:
            ```powershell
            [System.Environment]::SetEnvironmentVariable("JWT_SECRET_KEY", "your_generated_strong_secret_key_here", "Machine")
            [System.Environment]::SetEnvironmentVariable("STRATEGIES_DIR", "C:\TradingPlatformData\Strategies", "Machine")
            ```
        *   For user-specific variables (less common for services):
            ```powershell
            [System.Environment]::SetEnvironmentVariable("VARIABLE_NAME", "your_value", "User")
            ```
        *   **A system restart or at least a new login/service restart might be required** for these changes to be picked up by all processes, especially services.

*   **Security & Best Practices:**
    *   **NEVER** use default placeholder values from `.env.example` (like `"YOUR_PRODUCTION_..._CHANGE_ME"`) in a production environment.
    *   Ensure `JWT_SECRET_KEY`, `API_ENCRYPTION_KEY`, database passwords, and any other secrets are strong, unique, and kept confidential.
    *   The `STRATEGIES_DIR` must be an absolute path and secured appropriately.
    *   **`PYTHONUTF8=1`:** For Windows environments, it's sometimes necessary to set the `PYTHONUTF8=1` system environment variable to ensure Python handles UTF-8 encoding correctly for console output, file I/O, and subprocesses. If you encounter encoding-related errors, try setting this variable.

### 3.5. Database Migrations

Database migrations manage changes to your database schema. This project uses Alembic for migrations.

1.  **Ensure Prerequisites:**
    *   The Python virtual environment (`venv`) must be activated.
    *   All necessary environment variables, especially `DATABASE_URL`, must be set and accessible in your current Command Prompt or PowerShell session. If you set them via the System Properties GUI, you might need to open a new shell for them to take effect.
    *   You should be in the directory where Alembic is configured to run, typically the `backend` directory of your project (e.g., `C:\Projects\trading_platform\backend`).

2.  **Run Migrations:**
    *   Execute the following command to apply all pending migrations up to the latest version (`head`):
        ```shell
        alembic upgrade head
        ```
    *   **Explanation:**
        *   If this is a fresh database setup, this command will create all tables, indexes, and relationships as defined in your project's database models and Alembic migration scripts.
        *   If you are updating an existing deployment and new migration scripts have been added to the project, this command will apply those new changes to your database schema.
    *   Look for output indicating successful application of migrations (e.g., `INFO  [alembic.runtime.migration] Running upgrade ... -> ... OK`).

### 3.6. Running the FastAPI Application (as a service)

The FastAPI backend application is typically served using an ASGI server like Uvicorn. For production, it's recommended to run this as a Windows service.

1.  **Test Manually (Optional but Recommended):**
    *   Activate your virtual environment: `C:\Projects\trading_platform\backend\venv\Scripts\activate`
    *   Navigate to your project's root or backend directory (where your FastAPI `main.py` or equivalent is located). Assume it's `C:\Projects\trading_platform`.
    *   Run Uvicorn directly from the command line to ensure it starts correctly. Adjust `backend.main:app` if your main FastAPI app instance is located elsewhere.
        ```shell
        uvicorn backend.main:app --host 0.0.0.0 --port 8000
        ```
        (The `--host 0.0.0.0` makes it listen on all available network interfaces, which is common for services. The port `8000` is a common default; your reverse proxy will forward to this.)
    *   Try accessing `http://127.0.0.1:8000/docs` in a web browser on the server to see the FastAPI documentation. Press `Ctrl+C` to stop the manual run.

2.  **Setting up with NSSM:**
    *   Open Command Prompt or PowerShell **as Administrator**.
    *   Run the NSSM installation command:
        ```shell
        nssm install TradingPlatformAPI
        ```
        (You can choose a different service name if you prefer, e.g., `FastAPIService`).
    *   This opens the NSSM GUI. Configure the following:
        *   **`Application` Tab:**
            *   **Path:** Full path to the `python.exe` executable within your project's virtual environment.
                *   Example: `C:\Projects\trading_platform\backend\venv\Scripts\python.exe`
            *   **Startup directory:** Full path to your project's root directory (the one containing the `backend` folder, or directly the `backend` folder if your `main.py` expects to be run from there).
                *   Example: `C:\Projects\trading_platform` (if `main.py` is `backend/main.py`)
            *   **Arguments:** The arguments to pass to `python.exe`. This will be `-m uvicorn` followed by the application path and Uvicorn options.
                *   Example: `-m uvicorn backend.main:app --host 0.0.0.0 --port 8000`
                *   Adjust `backend.main:app` if your FastAPI application instance is named differently or located in a different file.
        *   **`Details` Tab:**
            *   **Display name:** e.g., `Trading Platform API Service`
            *   **Description:** e.g., `Handles backend API requests for the Trading Platform.`
        *   **`I/O` Tab (Recommended):**
            *   **Output (stdout):** Path to a log file, e.g., `C:\Projects\trading_platform\logs\api_stdout.log`. Create the `logs` directory if it doesn't exist.
            *   **Error (stderr):** Path to an error log file, e.g., `C:\Projects\trading_platform\logs\api_stderr.log`.
        *   **`Environment` Tab (CRUCIAL):**
            *   NSSM services sometimes do not reliably inherit all system or user environment variables, or may not pick up changes without a system reboot. For maximum reliability, it's best to define all required environment variables directly for the service.
            *   Add each variable on a new line in the format `VARIABLE_NAME=value`. Example:
                ```
                DATABASE_URL=postgresql://trading_platform_user:your_password@localhost/trading_platform_db
                JWT_SECRET_KEY=your_super_secret_key
                REDIS_URL=redis://localhost:6379/0
                STRATEGIES_DIR=C:\TradingPlatformData\Strategies
                ENVIRONMENT=production
                PYTHONUTF8=1
                ```
            *   Ensure all variables from your `.env.example` (and any others needed for production) are set here.
        *   **`Restart` Tab:**
            *   **Application exit:** Select "Restart application".
            *   You can configure delays and throttling if needed, but defaults are often fine to start.
        *   Click **"Install service"**.

3.  **Starting and Verifying the Service:**
    *   Start the service:
        ```shell
        nssm start TradingPlatformAPI
        ```
        Or use the Windows Services console (`services.msc`), find your service by its display name, and start it.
    *   **Verification:**
        *   Check `services.msc` to ensure the service status is "Running".
        *   Check the stdout/stderr log files you configured in NSSM for any startup errors.
        *   Try accessing the API locally (e.g., `http://127.0.0.1:8000/docs` or a health check endpoint if you have one). This will be before Nginx/IIS is configured to proxy it.

### 3.7. Running Celery Workers (as services)

Celery workers process background tasks. Running them as services is essential for production.

*   **Challenges on Windows:**
    *   Celery's support for Windows is limited. The default prefork pool is not available.
    *   It's **strongly recommended** to use the `solo` pool for simplicity and better stability on Windows: `celery -A your_app worker -P solo`.
    *   Alternatively, `eventlet` or `gevent` can be used if you install them (`pip install eventlet gevent`), but they might have their own complexities or compatibility issues on Windows. This guide will focus on `solo`.

*   **Environment Variables for Celery:**
    *   Celery workers will need access to the same environment variables as the FastAPI application (e.g., `DATABASE_URL`, `REDIS_URL`, `SMTP_HOST`, etc.) as tasks might interact with the database or other services.

*   **Setting up Main Celery Worker with NSSM:**
    1.  Open Command Prompt or PowerShell **as Administrator**.
    2.  Run the NSSM installation command:
        ```shell
        nssm install TradingPlatformCeleryWorker
        ```
    3.  **NSSM GUI Configuration:**
        *   **`Application` Tab:**
            *   **Path:** Full path to `python.exe` in your venv (e.g., `C:\Projects\trading_platform\backend\venv\Scripts\python.exe`).
            *   **Startup directory:** Full path to your project's root directory (e.g., `C:\Projects\trading_platform`).
            *   **Arguments:** `-m celery -A backend.celery_app.celery_app worker -l info -P solo -Q default_queue`
                *   Adjust `-A backend.celery_app.celery_app` if your Celery app instance is located elsewhere.
                *   `-l info` sets the log level to info.
                *   `-P solo` specifies the solo pool for Windows compatibility.
                *   `-Q default_queue` specifies the queue to consume from. You might have multiple workers for different queues.
        *   **`Details` Tab:**
            *   **Display name:** e.g., `Trading Platform Celery Worker (Default Queue)`
            *   **Description:** e.g., `Processes background tasks for the Trading Platform.`
        *   **`I/O` Tab:** Redirect stdout/stderr to log files (e.g., `C:\Projects\trading_platform\logs\celery_worker_default_stdout.log`).
        *   **`Environment` Tab:** **Crucial.** Set all necessary environment variables as described for the FastAPI service (copy them over).
        *   **`Restart` Tab:** Configure auto-restart.
        *   Click **"Install service"**.
    4.  Start the service: `nssm start TradingPlatformCeleryWorker`.

*   **Setting up Celery Beat Service (for scheduled tasks):**
    *   If your application uses scheduled tasks (e.g., the `auto_stop_expired_subscriptions` task, or any tasks defined in `celery_app.conf.beat_schedule`), you need to run a Celery Beat service. **Only one Celery Beat service should be running for your application.**
    1.  Test manually (activate venv, navigate to project root):
        ```shell
        celery -A backend.celery_app beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler
        ```
        (Use `--scheduler django_celery_beat.schedulers:DatabaseScheduler` if you are using `django-celery-beat` for storing schedules in the database. If your schedules are defined directly in `celery_app.conf.beat_schedule`, you can omit the `--scheduler` argument or use the default.)
    2.  Run NSSM installation for Celery Beat:
        ```shell
        nssm install TradingPlatformCeleryBeat
        ```
    3.  **NSSM GUI Configuration:**
        *   **`Application` Tab:**
            *   **Path:** `C:\Projects\trading_platform\backend\venv\Scripts\python.exe`
            *   **Startup directory:** `C:\Projects\trading_platform`
            *   **Arguments:** `-m celery -A backend.celery_app beat -l info --scheduler django_celery_beat.schedulers:DatabaseScheduler` (adjust scheduler if not using DB-backed schedules).
        *   **`Details` Tab:** Display name (e.g., `Trading Platform Celery Beat Scheduler`), Description.
        *   **`I/O` Tab:** Redirect to log files (e.g., `logs\celery_beat_stdout.log`).
        *   **`Environment` Tab:** Set all necessary environment variables.
        *   **`Restart` Tab:** Configure auto-restart.
        *   Click **"Install service"**.
    4.  Start the service: `nssm start TradingPlatformCeleryBeat`.

*   **Verification:**
    *   Check `services.msc` to ensure Celery worker and beat services are "Running".
    *   Monitor the log files you configured for any errors.
    *   Test by triggering an action that uses a Celery task (e.g., registering a new user should trigger the `send_email_task`). If you have scheduled tasks, check if they are being triggered according to their schedule by observing logs or their effects.

## 4. Frontend Setup

This section covers the deployment of the frontend application.

### 4.1. Copy Frontend Files

Instructions on where to place the built/static frontend files on the server.

### 4.2. Configure Web Server (Static Files & Reverse Proxy)

Detailed configuration examples for the chosen web server (e.g., Nginx) to:
- Serve the static frontend files (HTML, CSS, JS).
- Act as a reverse proxy for the backend FastAPI application, forwarding API requests (e.g., `/api/v1/*`) to the backend service.

## 5. Firewall Configuration

Guidance on configuring the Windows Firewall to allow inbound traffic on necessary ports (e.g., HTTP/80, HTTPS/443).

## 6. SSL Certificate Setup

Instructions and recommendations for obtaining and configuring an SSL certificate for HTTPS, ensuring secure communication. This may involve tools like Certbot (if applicable on Windows) or manual certificate installation.

## 7. Application Maintenance & Updates

Brief overview of common maintenance tasks, such as:
- Updating the application code from the repository.
- Re-running database migrations after updates.
- Restarting services.
- Monitoring logs.

## 8. Troubleshooting

Common issues and their solutions related to Windows deployment, such as service failures, path issues, permission problems, or connectivity errors.
