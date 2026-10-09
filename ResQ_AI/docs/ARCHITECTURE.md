# Architecture

The service is organized into preprocessing, safety, prediction, dispatch, and API layers. The design keeps rule-based safety logic separate from model-style predictions while exposing a unified process-emergency endpoint for downstream integrations.
