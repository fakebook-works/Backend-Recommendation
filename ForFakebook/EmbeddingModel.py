from __future__ import annotations

import hmac
import os
import uuid
from enum import Enum

import strawberry
from fastapi import Depends, FastAPI, Header, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from graphql import GraphQLError
from pydantic import BaseModel, Field, model_validator
from strawberry.fastapi import GraphQLRouter
from strawberry.types import Info

from .operations import (
    InteractionTargetUnavailableError,
    RecommendationOperations,
    RecommendationUnavailableError,
    get_operations,
)


GATEWAY_SECRET_HEADER = "X-Gateway-Secret"
RECOMMENDATION_INTERNAL_SECRET_HEADER = "X-Internal-RecommendationService-Secret"
CORRELATION_HEADER = "X-Correlation-ID"
USER_ID_HEADER = "X-User-Id"
MAX_SIGNED_64_BIT_ID = 9_223_372_036_854_775_807

app = FastAPI(title="Fakebook Recommendation", version="1.0")


@app.middleware("http")
async def internal_security_and_correlation(request: Request, call_next):
    correlation_id = request.headers.get(CORRELATION_HEADER) or uuid.uuid4().hex

    if request.url.path == "/internal" or request.url.path.startswith("/internal/"):
        expected_secret = os.getenv("RECOMMENDATION_INTERNAL_SECRET", "")
        provided_secret = request.headers.get(RECOMMENDATION_INTERNAL_SECRET_HEADER, "")
        if len(expected_secret.encode("utf-8")) < 32:
            response = JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "error": {
                        "code": "RECOMMENDATION_AUTH_NOT_CONFIGURED",
                        "message": "Recommendation internal authentication is not configured.",
                    }
                },
            )
            response.headers[CORRELATION_HEADER] = correlation_id
            return response

        if not hmac.compare_digest(
            expected_secret.encode("utf-8"),
            provided_secret.encode("utf-8"),
        ):
            response = JSONResponse(
                status_code=status.HTTP_403_FORBIDDEN,
                content={
                    "error": {
                        "code": "FORBIDDEN",
                        "message": "Internal service authentication failed.",
                    }
                },
            )
            response.headers[CORRELATION_HEADER] = correlation_id
            return response

    response = await call_next(request)
    response.headers[CORRELATION_HEADER] = correlation_id
    return response


class PostEmbeddingRequest(BaseModel):
    content: str = Field(default="", max_length=50_000)
    mediaUrls: list[str] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def require_content_or_media(self):
        if not self.content.strip() and not any(url.strip() for url in self.mediaUrls):
            raise ValueError("content or at least one media URL is required")
        return self


class UserEmbeddingPayload(BaseModel):
    success: bool
    userId: int
    created: bool
    message: str


class PostEmbeddingPayload(BaseModel):
    success: bool
    postId: int


class RecommendationInteractionAction(str, Enum):
    LIKE = "LIKE"
    UNLIKE = "UNLIKE"
    SAVE = "SAVE"
    UNSAVE = "UNSAVE"
    WATCH = "WATCH"
    SHARE = "SHARE"
    COMMENT = "COMMENT"


class RecommendationInteractionRequest(BaseModel):
    targetId: int = Field(gt=0, le=MAX_SIGNED_64_BIT_ID)
    action: RecommendationInteractionAction


class RecommendationInteractionPayload(BaseModel):
    success: bool
    applied: bool
    userId: int
    targetId: int
    action: RecommendationInteractionAction


def _translate_operation_error(exception: Exception) -> HTTPException:
    if isinstance(exception, InteractionTargetUnavailableError):
        return HTTPException(status_code=425, detail=str(exception))
    if isinstance(exception, RecommendationUnavailableError):
        return HTTPException(status_code=503, detail=str(exception))
    if isinstance(exception, ValueError):
        return HTTPException(status_code=400, detail=str(exception))
    return HTTPException(status_code=500, detail="Recommendation operation failed.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.put(
    "/internal/recommendation/users/{user_id}/embedding",
    response_model=UserEmbeddingPayload,
)
def ensure_user_embedding(
    user_id: int,
    operations: RecommendationOperations = Depends(get_operations),
) -> UserEmbeddingPayload:
    try:
        created = operations.ensure_user_embedding(user_id)
    except Exception as exception:
        raise _translate_operation_error(exception) from exception

    return UserEmbeddingPayload(
        success=True,
        userId=user_id,
        created=created,
        message="User embedding created." if created else "User embedding already exists.",
    )


@app.delete(
    "/internal/recommendation/users/{user_id}/embedding",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_user_embedding(
    user_id: int,
    operations: RecommendationOperations = Depends(get_operations),
) -> Response:
    try:
        operations.delete_user_embedding(user_id)
    except Exception as exception:
        raise _translate_operation_error(exception) from exception
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.put(
    "/internal/recommendation/posts/{post_id}/embedding",
    response_model=PostEmbeddingPayload,
)
def upsert_post_embedding(
    post_id: int,
    request: PostEmbeddingRequest,
    operations: RecommendationOperations = Depends(get_operations),
) -> PostEmbeddingPayload:
    try:
        operations.upsert_post_embedding(post_id, request.content, request.mediaUrls)
    except Exception as exception:
        raise _translate_operation_error(exception) from exception
    return PostEmbeddingPayload(success=True, postId=post_id)


@app.delete(
    "/internal/recommendation/posts/{post_id}/embedding",
    status_code=status.HTTP_204_NO_CONTENT,
)
def delete_post_embedding(
    post_id: int,
    operations: RecommendationOperations = Depends(get_operations),
) -> Response:
    try:
        operations.delete_post_embedding(post_id)
    except Exception as exception:
        raise _translate_operation_error(exception) from exception
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@app.post(
    "/internal/recommendation/users/{user_id}/interactions",
    response_model=RecommendationInteractionPayload,
)
def record_recommendation_interaction(
    user_id: int,
    request: RecommendationInteractionRequest,
    idempotency_key: str = Header(
        ...,
        alias="Idempotency-Key",
        min_length=1,
        max_length=128,
    ),
    operations: RecommendationOperations = Depends(get_operations),
) -> RecommendationInteractionPayload:
    try:
        applied = operations.record_interaction(
            user_id,
            request.targetId,
            request.action.value,
            idempotency_key,
        )
    except Exception as exception:
        raise _translate_operation_error(exception) from exception

    return RecommendationInteractionPayload(
        success=True,
        applied=applied,
        userId=user_id,
        targetId=request.targetId,
        action=request.action,
    )


@strawberry.type
class RecommendationItem:
    post_id: strawberry.ID


@strawberry.type
class ReelRecommendationItem:
    reel_id: strawberry.ID


@strawberry.enum
class ReelRecommendationMode(Enum):
    FOR_YOU = "FOR_YOU"
    FOLLOWING = "FOLLOWING"


@strawberry.type
class Query:
    @strawberry.field
    def hello(self) -> str:
        return "Hello from Fakebook Recommendation"

    @strawberry.field
    def recommend_feed(
        self,
        info: Info,
        user_id: strawberry.ID,
        skip: int = 0,
        take: int = 20,
    ) -> list[RecommendationItem]:
        parsed_user_id = _parse_user_id(user_id)
        _require_trusted_viewer(info, parsed_user_id)
        operations: RecommendationOperations = info.context["operations"]
        rows = operations.recommend_feed(
            parsed_user_id,
            skip,
            take,
            info.context.get("correlation_id"),
        )
        return [
            RecommendationItem(
                post_id=strawberry.ID(str(row["postId"])),
            )
            for row in rows
        ]

    @strawberry.field
    def recommend_reels(
        self,
        info: Info,
        user_id: strawberry.ID,
        mode: ReelRecommendationMode = ReelRecommendationMode.FOR_YOU,
        skip: int = 0,
        take: int = 20,
    ) -> list[ReelRecommendationItem]:
        parsed_user_id = _parse_user_id(user_id)
        _require_trusted_viewer(info, parsed_user_id)
        operations: RecommendationOperations = info.context["operations"]
        rows = operations.recommend_reels(
            parsed_user_id,
            mode.value,
            skip,
            take,
            info.context.get("correlation_id"),
        )
        return [
            ReelRecommendationItem(
                reel_id=strawberry.ID(str(row["reelId"])),
            )
            for row in rows
        ]


def _parse_user_id(user_id: strawberry.ID) -> int:
    try:
        parsed_user_id = int(str(user_id))
    except ValueError as exception:
        raise GraphQLError(
            "userId must be a positive signed 64-bit integer.",
            extensions={"code": "BAD_USER_INPUT"},
        ) from exception

    if parsed_user_id <= 0 or parsed_user_id > MAX_SIGNED_64_BIT_ID:
        raise GraphQLError(
            "userId must be a positive signed 64-bit integer.",
            extensions={"code": "BAD_USER_INPUT"},
        )
    return parsed_user_id


def _require_trusted_viewer(info: Info, requested_user_id: int) -> None:
    request: Request = info.context["request"]
    expected_secret = os.getenv("INTERNAL_SHARED_SECRET", "")
    if len(expected_secret.encode("utf-8")) < 32:
        raise GraphQLError(
            "Recommendation trusted caller authentication is not configured.",
            extensions={"code": "SERVICE_UNAVAILABLE"},
        )

    provided_secret = request.headers.get(GATEWAY_SECRET_HEADER, "")
    if not hmac.compare_digest(
        expected_secret.encode("utf-8"),
        provided_secret.encode("utf-8"),
    ):
        raise GraphQLError(
            "Trusted Gateway authentication failed.",
            extensions={"code": "FORBIDDEN"},
        )

    trusted_user_id = request.headers.get(USER_ID_HEADER, "")
    if not trusted_user_id:
        raise GraphQLError(
            "Authentication is required.",
            extensions={"code": "UNAUTHENTICATED"},
        )
    try:
        parsed_user_id = int(trusted_user_id)
    except ValueError as exception:
        raise GraphQLError(
            "Trusted user identity is invalid.",
            extensions={"code": "FORBIDDEN"},
        ) from exception

    if parsed_user_id != requested_user_id:
        raise GraphQLError(
            "Requested user does not match the authenticated user.",
            extensions={"code": "FORBIDDEN"},
        )


async def graphql_context(
    request: Request,
    operations: RecommendationOperations = Depends(get_operations),
) -> dict:
    return {
        "operations": operations,
        "correlation_id": request.headers.get(CORRELATION_HEADER),
        "request": request,
    }


schema = strawberry.Schema(query=Query)
graphql_app = GraphQLRouter(schema, context_getter=graphql_context)
app.include_router(graphql_app, prefix="/graphql")
