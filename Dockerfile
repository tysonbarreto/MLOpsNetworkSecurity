ARG PYTHON_BASE=3.11-slim

FROM python:$PYTHON_BASE as builder

RUN pip install -U pdm
ENV PDM_CHECK_UPDATE=false

COPY pyproject.toml pdm.lock README.md /project/
COPY src/ /project/src
# COPY .pdm-build/ /project/.pdm-build
# COPY dist/ /project/dist


WORKDIR /project
RUN pdm install --check --prod --no-editable

FROM python:$PYTHON_BASE
COPY --from=builder /project/.venv/ /project/.venv
ENV PATH="/project/.venv/bin:$PATH"
COPY src /project/src
COPY app.py /project/
COPY pipeline /project/pipeline
COPY schema /project/schema
COPY templates /project/templates
COPY .env /project/
WORKDIR /project
CMD ["python", "/project/app.py"]