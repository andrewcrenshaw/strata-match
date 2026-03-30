"""Shared test fixtures for strata-match."""

from __future__ import annotations

import numpy as np
import pytest

from strata_match.embeddings import EmbeddingProvider
from strata_match.models import CandidateProfile, JobDescription

from numpy.typing import NDArray


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic embedding provider for tests.

    Produces consistent embeddings based on text hash so tests are reproducible.
    """

    def __init__(self, dimension: int = 8) -> None:
        self._dimension = dimension

    async def embed(self, text: str) -> NDArray[np.float32]:
        rng = np.random.default_rng(seed=hash(text) % (2**31))
        vec = rng.standard_normal(self._dimension).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    async def embed_batch(self, texts: list[str]) -> list[NDArray[np.float32]]:
        return [await self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dimension


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimension=8)


@pytest.fixture
def sample_profile() -> CandidateProfile:
    return CandidateProfile(
        title="Senior Software Engineer",
        skills=["Python", "FastAPI", "PostgreSQL", "AWS", "Docker"],
        experience_years=8,
        summary="Full-stack engineer focused on distributed systems and API design.",
        education=["BS Computer Science"],
        industries=["SaaS", "FinTech"],
    )


@pytest.fixture
def sample_jobs() -> list[JobDescription]:
    return [
        JobDescription(
            title="Staff Engineer — Backend Platform",
            company="Acme Corp",
            description="Lead our backend platform team building scalable APIs.",
            requirements=["Python", "System Design", "PostgreSQL", "Leadership"],
            location="Remote",
            salary_range="$180k-$220k",
        ),
        JobDescription(
            title="Junior Frontend Developer",
            company="WebCo",
            description="Build React components for our marketing site.",
            requirements=["React", "CSS", "HTML", "JavaScript"],
            location="New York, NY",
        ),
        JobDescription(
            title="Data Scientist",
            company="DataLabs",
            description="Build ML models for customer churn prediction.",
            requirements=["Python", "scikit-learn", "SQL", "Statistics"],
            preferred_qualifications=["PhD", "TensorFlow"],
            location="San Francisco, CA",
        ),
    ]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
