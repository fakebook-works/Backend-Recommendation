from __future__ import annotations

import os

from fastapi import FastAPI
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased


def configure_observability(app: FastAPI, service_name: str) -> None:
    """Export traces/metrics without collecting bodies, variables, or auth headers."""

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.instance.id": os.getenv("HOSTNAME", "local"),
            }
        )
        try:
            ratio = min(1.0, max(0.0, float(os.getenv("OTEL_TRACES_SAMPLER_ARG", "0.1"))))
        except ValueError:
            ratio = 0.1

        tracer_provider = TracerProvider(
            resource=resource,
            sampler=ParentBased(TraceIdRatioBased(ratio)),
        )
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=endpoint.startswith("http://")))
        )
        trace.set_tracer_provider(tracer_provider)

        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=endpoint.startswith("http://"))
        )
        metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[metric_reader]))

    FastAPIInstrumentor.instrument_app(app, excluded_urls="health/live,health/ready")
    RequestsInstrumentor().instrument()
