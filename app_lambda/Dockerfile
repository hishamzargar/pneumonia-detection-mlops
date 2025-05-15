# Use an AWS-provided base image for Python 3.9 on arm64 architecture
#FROM public.ecr.aws/lambda/python:3.11-arm64
FROM public.ecr.aws/lambda/python:3.11-x86_64

COPY app.py ${LAMBDA_TASK_ROOT}/
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Test basic ONNX Runtime import and initialization right after installation
RUN echo "Attempting to import numpy..." \
    && python -c "import numpy; print(f'NumPy version: {numpy.__version__}')" \
    && echo "Attempting to import Pillow..." \
    && python -c "import PIL; print(f'Pillow version: {PIL.__version__}')" \
    && echo "Attempting to import onnxruntime..." \
    && python -c "import onnxruntime; print(f'ONNX Runtime version: {onnxruntime.__version__}')" \
    && echo "Attempting to get ONNX Runtime providers..." \
    && python -c "import onnxruntime; print(f'Available providers: {onnxruntime.get_available_providers()}')" \
    && echo "All Python import and ONNX Runtime initialization tests passed successfully in Docker build."

CMD [ "app.lambda_handler" ]