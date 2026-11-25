FROM node:22-bookworm-slim

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY *.mjs ./
COPY public ./public
COPY model_yolo_nano ./model_yolo_nano
COPY model_yolo ./model_yolo

ENV MODEL_BACKEND=yolo
ENV MODEL_FOLDER=model_yolo_nano
ENV EXECUTION_LOCATION=cloud
ENV ORT_NUM_THREADS=1

EXPOSE 10000

CMD ["npm", "start"]