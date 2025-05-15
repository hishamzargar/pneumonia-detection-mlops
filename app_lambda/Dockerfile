# Use an AWS-provided base image for Python 3.9 on arm64 architecture
#FROM public.ecr.aws/lambda/python:3.11-arm64
FROM public.ecr.aws/lambda/python:3.11-x86_64

COPY app.py ${LAMBDA_TASK_ROOT}/
COPY requirements_lambda.txt ${LAMBDA_TASK_ROOT}/

# Install dependencies
RUN pip install --no-cache-dir -r requirements_lambda.txt

# Test basic ONNX Runtime import and initialization right after installation
RUN python -c "import onnxruntime; print(f'Successfully imported ONNX Runtime version: {onnxruntime.__version__}'); print(f'Available providers: {onnxruntime.get_available_providers()}')"

CMD [ "app.lambda_handler" ]