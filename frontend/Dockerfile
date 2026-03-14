# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --silent

COPY . .

# frontend/.env.production is written by the Makefile/start.sh before
# docker build runs — it sets VITE_API_URL=/api/v1 so Vite bakes
# relative URLs into the bundle instead of hardcoded localhost ports.
RUN npm run build


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM nginx:1.27-alpine AS runtime

RUN rm -rf /usr/share/nginx/html/*

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# map.conf must load before default.conf — alphabetical order guarantees this.
# It defines the $connection_upgrade map used for WebSocket handling.
COPY map.conf     /etc/nginx/conf.d/map.conf
COPY nginx.conf   /etc/nginx/conf.d/default.conf

EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:3000/nginx-health || exit 1

CMD ["nginx", "-g", "daemon off;"]