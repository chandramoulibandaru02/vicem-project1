import logging
from datetime import datetime, timezone
from threading import Lock

from fastapi import HTTPException


class WorkflowService:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("ecm_ai_backend.workflow")
        self._lock = Lock()
        self._workflows: dict[str, dict[str, object]] = {}

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

    def _build_record(self, document_id: str, status: str, approver: str | None = None) -> dict[str, object]:
        return {
            "document_id": document_id,
            "status": status,
            "approver": approver,
            "created_at": self._utc_now(),
            "updated_at": self._utc_now(),
        }

    def create_workflow(self, document_id: str, approver: str | None = None) -> dict[str, object]:
        with self._lock:
            if document_id in self._workflows:
                existing = self._workflows[document_id]
                self.logger.warning("Workflow already exists", extra={"document_id": document_id, "status": existing["status"]})
                raise HTTPException(status_code=409, detail="Workflow already exists for this document")

            workflow = self._build_record(document_id, "pending", approver)
            self._workflows[document_id] = workflow

        self.logger.info("Created workflow", extra={"document_id": document_id, "status": "pending", "approver": approver})
        return workflow

    def approve_workflow(self, document_id: str, approver: str | None = None) -> dict[str, object]:
        with self._lock:
            workflow = self._workflows.get(document_id)
            if workflow is None:
                self.logger.warning("Workflow approve requested for missing document", extra={"document_id": document_id})
                raise HTTPException(status_code=404, detail="Workflow not found")

            workflow["status"] = "approved"
            workflow["approver"] = approver
            workflow["updated_at"] = self._utc_now()

        self.logger.info("Approved workflow", extra={"document_id": document_id, "approver": approver})
        return workflow

    def reject_workflow(self, document_id: str, approver: str | None = None) -> dict[str, object]:
        with self._lock:
            workflow = self._workflows.get(document_id)
            if workflow is None:
                self.logger.warning("Workflow reject requested for missing document", extra={"document_id": document_id})
                raise HTTPException(status_code=404, detail="Workflow not found")

            workflow["status"] = "rejected"
            workflow["approver"] = approver
            workflow["updated_at"] = self._utc_now()

        self.logger.info("Rejected workflow", extra={"document_id": document_id, "approver": approver})
        return workflow

    def get_workflow_status(self, document_id: str) -> dict[str, object]:
        with self._lock:
            workflow = self._workflows.get(document_id)
            if workflow is None:
                self.logger.warning("Workflow status requested for missing document", extra={"document_id": document_id})
                raise HTTPException(status_code=404, detail="Workflow not found")

        return workflow
