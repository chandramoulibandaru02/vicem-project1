from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.services.workflow_service import WorkflowService

router = APIRouter(prefix="/workflow", tags=["workflow"])
workflow_service = WorkflowService()


class WorkflowCreateRequest(BaseModel):
    document_id: str = Field(..., min_length=1, description="Document identifier for the workflow")
    approver: str | None = Field(default=None, description="Optional approver name")


class WorkflowActionRequest(BaseModel):
    approver: str | None = Field(default=None, description="Optional approver name")


class WorkflowResponse(BaseModel):
    document_id: str
    status: str
    approver: str | None
    created_at: str
    updated_at: str


@router.post("", response_model=WorkflowResponse, summary="Create a workflow for a document")
async def create_workflow(payload: WorkflowCreateRequest):
    if not payload.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id is required")

    workflow = workflow_service.create_workflow(payload.document_id, payload.approver)
    return WorkflowResponse(**workflow)


@router.post("/{document_id}/approve", response_model=WorkflowResponse, summary="Approve a workflow")
async def approve_workflow(document_id: str, payload: WorkflowActionRequest):
    workflow = workflow_service.approve_workflow(document_id, payload.approver)
    return WorkflowResponse(**workflow)


@router.post("/{document_id}/reject", response_model=WorkflowResponse, summary="Reject a workflow")
async def reject_workflow(document_id: str, payload: WorkflowActionRequest):
    workflow = workflow_service.reject_workflow(document_id, payload.approver)
    return WorkflowResponse(**workflow)


@router.get("/{document_id}", response_model=WorkflowResponse, summary="Get workflow status")
async def get_workflow_status(document_id: str):
    workflow = workflow_service.get_workflow_status(document_id)
    return WorkflowResponse(**workflow)
