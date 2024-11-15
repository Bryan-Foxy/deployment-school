FROM python:3.12-slim
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /website
COPY . .
RUN chmod -R 755 /website/static
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gdown && \
	gdown --id 1_D96kmghOcwlK80PN73kvA1ad5d6cAUF -O /website/models/sam_vit_h_4b8939.pth

EXPOSE 5400
CMD ["python", "setup.py"]