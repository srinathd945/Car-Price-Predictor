FROM python:3.8

# Set working directory
WORKDIR /opt/car_price

# Copy requirement file and install dependencies
COPY requirement.txt .
RUN pip3 install -r requirement.txt

# Create templates directory
RUN mkdir templates

# Copy data and model training script
COPY car data.csv .
COPY model.py .

# Run model training
RUN python3.8 model.py

# Copy templates and Flask app
COPY templates/index.html templates
COPY templates/result.html templates
COPY flaskapp.py .

# Run the Flask application
ENTRYPOINT ["python3.8", "flaskapp.py"]
