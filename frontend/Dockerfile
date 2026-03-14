# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# Install node deps and build the Vite/React app into /app/dist
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests first for layer caching
COPY package.json package-lock.json ./

RUN npm ci --silent

# Copy the rest of the source
COPY . .

# Build production bundle — outputs to /app/dist
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# Serve the static bundle with nginx
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

# Remove the default nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy built assets from builder
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom nginx config that:
#   - Serves the SPA (all routes → index.html)
#   - Proxies /api/* to the backend API
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:80/ || exit 1

CMD ["nginx", "-g", "daemon off;"]