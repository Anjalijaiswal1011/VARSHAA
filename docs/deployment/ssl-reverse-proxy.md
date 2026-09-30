# Production Reverse Proxy, SSL Termination & Domain Routing Guide

**System:** VARSHAA (RAIN-REPAIR X)  
**Task ID:** P8-02  
**Protocol:** HTTPS / TLS 1.3 / HTTP/2  

---

## 1. Domain & DNS Configuration

Configure DNS records with your registrar or Cloudflare/Route53:

| Record Type | Host | Points To | TTL | Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `varshaa.imd.gov.in` | `<Public Elastic IP>` | 300 | Production National Portal |
| **A** | `api.varshaa.imd.gov.in` | `<Public Elastic IP>` | 300 | High-throughput Machine API |
| **CNAME** | `demo.varshaa.ai` | `varshaa.imd.gov.in` | 300 | SIH Live Evaluation Sandbox |

---

## 2. Automated TLS Certificate Provisioning (Certbot)

Run Let's Encrypt automated challenge:
```bash
certbot --nginx -d varshaa.imd.gov.in -d demo.varshaa.ai \
    --agree-tos --email ops@varshaa.imd.gov.in --non-interactive
```

Verify automated renewal cron:
```bash
systemctl list-timers | grep certbot
# Test dry run renewal:
certbot renew --dry-run
```

---

## 3. Reverse Proxy Architecture

```
                       Internet (Client / IMD Forecaster / NDMA)
                                         │
                                         ▼ [Port 443 HTTPS / TLS 1.3]
                        ┌─────────────────────────────────┐
                        │          Nginx Ingress          │
                        │  - SSL Termination              │
                        │  - Rate Limiting (100 req/min)  │
                        │  - Gzip / Brotli Compression    │
                        │  - HSTS & Security Headers      │
                        └────────────────┬────────────────┘
                                         │
                ┌────────────────────────┴────────────────────────┐
                │                                                 │
        [Static Routing]                                  [Dynamic Reverse Proxy]
                │                                                 │
                ▼                                                 ▼
     /usr/share/nginx/html                              http://127.0.0.1:8000
 (React Dashboard / Assets)                               (Uvicorn / FastAPI)
   - /dashboard                                             - /api/v1/forecasts/*
   - /index.html                                            - /api/v1/health
   - /assets/*.js, *.css                                    - /api/v1/explainability/*
```

---

## 4. Rate Limiting & DDoS Safeguards

Add to `/etc/nginx/nginx.conf`:
```nginx
# IP-based rate limiting zone
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=60r/m;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

# Within location /api/ block:
location /api/ {
    limit_req zone=api_limit burst=20 nodelay;
    limit_conn conn_limit 10;
    proxy_pass http://backend_cluster;
}
```

---

## 5. SSL Security Audit & Grades

- **Qualys SSL Labs Target:** A+ Rating
- **TLS Protocols:** Strict TLS 1.2 and TLS 1.3 only; SSLv2, SSLv3, TLS 1.0, TLS 1.1 disabled.
- **Strict-Transport-Security:** `max-age=15768000; includeSubDomains; preload`
- **X-Content-Type-Options:** `nosniff`
- **X-Frame-Options:** `SAMEORIGIN`
