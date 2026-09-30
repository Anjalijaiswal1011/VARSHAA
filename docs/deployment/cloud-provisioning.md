# Production Cloud Provisioning & Storage Architecture Guide

**System:** VARSHAA — RAAP-X Operational Precipitation Post-Processing Engine  
**Target Environments:** AWS EC2, Azure VM, GCP Compute Engine, NIC National Cloud (MeghRaj)  
**Task ID:** P8-01  

---

## 1. Compute & Resource Sizing

| Component | Minimum Specification | Recommended Production | High-Availability Cluster |
| :--- | :--- | :--- | :--- |
| **Instance Type** | 4 vCPU, 16 GB RAM | 8 vCPU, 32 GB RAM (e.g. `c6i.2xlarge`) | $2\times$ 8 vCPU, 32 GB behind Load Balancer |
| **Operating System** | Ubuntu 22.04 LTS x86_64 | Ubuntu 24.04 LTS x86_64 | Ubuntu 24.04 LTS x86_64 |
| **Root Disk** | 50 GB SSD (ext4) | 100 GB NVMe SSD | 100 GB NVMe SSD |
| **Persistent Data Volume** | 100 GB mounted at `/mnt/varshaa_data` | 500 GB NVMe / EBS gp3 (3000 IOPS) | Distributed Ceph / AWS EFS mount |
| **Network** | 1 Gbps | Up to 12.5 Gbps Enhanced Networking | Dedicated VPC Peering with IMD GTS nodes |

---

## 2. Directory Hierarchy & Storage Layout

On the host VM:
```bash
/opt/varshaa/                      # Application repository root
├── backend/                       # FastAPI application
├── src/                           # Machine Learning & GIS pipeline
├── frontend/dist/                 # Static built React SPA dashboard
└── venv/                          # Isolated Python virtual environment

/mnt/varshaa_data/                 # High-speed persistent block storage
├── raw/                           # NetCDF4 / GRIB2 raw NWP feeds (GFS/NCUM)
├── interim/                       # Regridded 0.25° intermediate tensors
├── processed/                     # Feature-engineered Parquet files
├── models/                        # Versioned model weights (.joblib, .lgb)
│   └── registry/                  # Production metadata JSON registry
└── db/                            # SQLite / PostgreSQL operational database
    └── varshaa.db
```

---

## 3. Host Initialization Script

Execute the following bash commands during cloud VM bootstrapping (`cloud-init`):

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Update OS and install runtime C libraries
apt-get update && apt-get upgrade -y
apt-get install -y python3.11 python3.11-venv python3.11-dev \
    libgeos-dev libproj-dev libspatialindex-dev libgomp1 \
    nginx certbot python3-certbot-nginx curl git htop jq

# 2. Create isolated system service user
useradd -r -s /bin/false -d /opt/varshaa varshaa || true

# 3. Mount and configure persistent storage
mkdir -p /mnt/varshaa_data /etc/varshaa
chown -R varshaa:varshaa /mnt/varshaa_data

# 4. Clone and set up repository
git clone https://github.com/Anjalijaiswal1011/VARSHAA.git /opt/varshaa
cd /opt/varshaa
python3.11 -m venv venv
venv/bin/pip install --upgrade pip wheel
venv/bin/pip install -r requirements.txt

# 5. Build frontend SPA
cd /opt/varshaa/frontend
npm ci && npm run build
chown -R varshaa:varshaa /opt/varshaa

# 6. Install systemd service
cp /opt/varshaa/deployment/systemd/varshaa-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable --now varshaa-backend
```

---

## 4. Log Rotation & Disk Maintenance

Place `/etc/logrotate.d/varshaa`:
```conf
/opt/varshaa/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 varshaa varshaa
}
```

---

## 5. Disaster Recovery & Backup Cadence

1. **Database Snapshots**: Hourly SQLite WAL backup using `sqlite3 /mnt/varshaa_data/db/varshaa.db ".backup /mnt/varshaa_data/backups/varshaa_$(date +%s).bak"`.
2. **Model Registry**: Synced to immutable S3 bucket `s3://varshaa-model-registry-prod/` with Object Lock.
3. **Recovery Time Objective (RTO)**: $< 15$ minutes via pre-baked AMI / Docker container.
4. **Recovery Point Objective (RPO)**: $< 1$ hour.
