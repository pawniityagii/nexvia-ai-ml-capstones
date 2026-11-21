# Kubernetes (Minikube) deployment

> Status: manifests are validated with `kubectl apply --dry-run=client`. They
> are provided as the project's deployment target; a full live-cluster run is an
> optional extension, not claimed as a two-week production operation.

## Prerequisites
- `minikube` and `kubectl` installed
- The model trained so `artifacts/` exists

## Steps
```bash
minikube start --memory 4096 --cpus 2

# build images straight into minikube's docker
minikube image build -t fraud-api:latest .
minikube image build -t fraud-dashboard:latest -f dashboard/Dockerfile .

kubectl apply -f deploy/k8s/namespace.yaml
kubectl apply -f deploy/k8s/kafka.yaml
kubectl apply -f deploy/k8s/api.yaml
kubectl apply -f deploy/k8s/dashboard.yaml

kubectl -n fraud get pods
kubectl -n fraud port-forward svc/fraud-api 8000:8000      # API
minikube service fraud-dashboard -n fraud                  # dashboard URL
```

## Notes
- `artifacts/` (the trained model) is baked into the image at build time here.
  In a real cluster you would mount a PersistentVolume or pull the model from a
  registry/MLflow model store instead.
- Kafka runs single-node in KRaft mode - fine for the demo, not HA.
