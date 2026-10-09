"""Director Brain 数据模型包（Pydantic v2）。"""
from __future__ import annotations

from director_brain.models.base import BaseRecord
from director_brain.models.director_brief import DirectorBrief
from director_brain.models.director_plan import (
    Decision,
    DirectorDecisionPlan,
    ProjectNarrativeActBoundary,
    ProjectNarrativeConstraintAssessment,
    ProjectNarrativeEvidenceClaim,
    ProjectNarrativeEvidenceRef,
    ProjectNarrativeReasoningTrace,
    ProjectNarrativeShotHypothesis,
    ProjectNarrativeSourceRationale,
    ProjectNarrativeStrategyHypothesis,
    ProjectDirectorSourceSelectionAuditEntry,
    ProjectDirectorStrategyCandidate,
    ProjectDirectorShadowComparison,
)
from director_brain.models.edl import EditItem, EditorialDecisionList
from director_brain.models.film_context import (
    AssetAnalysisState,
    ContextLayer,
    FilmContextSnapshot,
    ObservationTimebase,
    ProjectAssetCoverage,
)
from director_brain.models.film_entity import EntityType, FilmEntity, StoryRelation
from director_brain.models.film_observation import (
    ClaimKind,
    FILM_OBSERVATION_SCHEMA_VERSION,
    FilmObservation,
    TimebaseUnit,
)
from director_brain.models.project import (
    FilmProjectManifest,
    ProjectAnalysisProfile,
    ProjectAsset,
    ProjectAssetTimeMap,
    ProjectBoundaryBasis,
    ProcessingRights,
    ProjectStreamTiming,
    RightsBasis,
    RightsState,
)
from director_brain.models.revision import RevisionProposal
from director_brain.models.story_graph import (
    StoryEdge,
    StoryEdgeType,
    StoryGraph,
    StoryNode,
    StoryNodeType,
)
from director_brain.models.project_story_graph import (
    ProjectAssetStoryGraph,
    ProjectStoryGraph,
    ProjectStoryGraphView,
)
from director_brain.models.project_story_link_review import (
    ProjectStoryLinkComparisonDisposition,
    ProjectStoryEntityLink,
    ProjectStoryLinkAnchor,
    ProjectStoryLinkReview,
)
from director_brain.models.project_story_link_comparison import (
    ProjectStoryLinkComparison,
    ProjectStoryLinkEventEvidence,
)
from director_brain.models.project_story_mention_review import (
    ProjectStoryMentionDecision,
    ProjectStoryMentionReview,
)
from director_brain.models.project_story_mention_preview import (
    ProjectStoryMentionPreview,
    ProjectStoryMentionPreviewFrame,
)
from director_brain.models.project_story_link_candidate_preview import (
    ProjectStoryLinkCandidatePreview,
)
from director_brain.models.project_story_link_candidates import (
    ProjectStoryLinkCandidate,
    ProjectStoryLinkCandidateAssetCoverage,
    ProjectStoryLinkCandidatePage,
    ProjectStoryLinkComparisonRequestData,
)
from director_brain.models.project_director_shadow_comparison import (
    ProjectDirectorShadowComparisonRecord,
)
from director_brain.models.project_context_job import (
    ProjectContextJob,
    ProjectContextJobProgress,
    ProjectContextJobState,
)

from director_brain.models.parameterization import (
    AudioEvent,
    AudioEventType,
    CandidateSource,
    ClipBoundary,
    FeasibleRange,
    ParameterCandidate,
    ParameterizationContext,
    ParameterizationDecision,
    ParameterizationStatus,
)

__all__ = [
    "BaseRecord",
    "DirectorBrief",
    "FilmContextSnapshot",
    "ContextLayer",
    "AssetAnalysisState",
    "ObservationTimebase",
    "ProjectAssetCoverage",
    "FilmObservation",
    "ClaimKind",
    "FILM_OBSERVATION_SCHEMA_VERSION",
    "TimebaseUnit",
    "FilmProjectManifest",
    "ProjectAnalysisProfile",
    "ProjectAsset",
    "ProjectAssetTimeMap",
    "ProjectBoundaryBasis",
    "ProcessingRights",
    "ProjectStreamTiming",
    "RightsBasis",
    "RightsState",
    "FilmEntity",
    "StoryRelation",
    "EntityType",
    "StoryGraph",
    "StoryNode",
    "StoryEdge",
    "StoryNodeType",
    "StoryEdgeType",
    "ProjectStoryGraph",
    "ProjectStoryGraphView",
    "ProjectAssetStoryGraph",
    "ProjectStoryLinkReview",
    "ProjectStoryLinkComparison",
    "ProjectStoryLinkEventEvidence",
    "ProjectStoryLinkCandidate",
    "ProjectStoryLinkCandidateAssetCoverage",
    "ProjectStoryLinkCandidatePage",
    "ProjectStoryLinkComparisonRequestData",
    "ProjectContextJob",
    "ProjectContextJobProgress",
    "ProjectContextJobState",
    "ProjectStoryMentionDecision",
    "ProjectStoryMentionReview",
    "ProjectStoryMentionPreview",
    "ProjectStoryMentionPreviewFrame",
    "ProjectStoryLinkCandidatePreview",
    "ProjectStoryLinkComparisonDisposition",
    "ProjectStoryEntityLink",
    "ProjectStoryLinkAnchor",
    "EditorialDecisionList",
    "EditItem",
    "DirectorDecisionPlan",
    "Decision",
    "ProjectNarrativeEvidenceRef",
    "ProjectNarrativeShotHypothesis",
    "ProjectNarrativeActBoundary",
    "ProjectNarrativeConstraintAssessment",
    "ProjectNarrativeEvidenceClaim",
    "ProjectNarrativeReasoningTrace",
    "ProjectNarrativeSourceRationale",
    "ProjectNarrativeStrategyHypothesis",
    "ProjectDirectorSourceSelectionAuditEntry",
    "ProjectDirectorStrategyCandidate",
    "ProjectDirectorShadowComparison",
    "ProjectDirectorShadowComparisonRecord",
    "RevisionProposal",
    "ParameterizationDecision",
    "ParameterizationContext",
    "ParameterizationStatus",
    "ParameterCandidate",
    "CandidateSource",
    "FeasibleRange",
    "ClipBoundary",
    "AudioEvent",
    "AudioEventType",
]
