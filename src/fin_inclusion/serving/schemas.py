"""Request/response contract for the prediction API.

Category values are spelled exactly as in the raw survey CSVs (including the
original typo "Divorced/Seperated" and the "Dont know"-style placeholders,
which the model treats as a real, informative answer -- see EDA 1.3). The
schema is strict on purpose: the training pipeline would quietly route an
unseen category like "uganda" to an all-zero one-hot row, which is exactly
the silent mis-scoring an API must refuse. The service cross-checks these
literals against the model artifact's vocabulary at startup
(`predictor.check_schema_matches_artifact`), so the two cannot drift apart
unnoticed.
"""
from __future__ import annotations

import typing
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Country = Literal["Kenya", "Rwanda", "Tanzania", "Uganda"]
LocationType = Literal["Rural", "Urban"]
YesNo = Literal["Yes", "No"]
Gender = Literal["Female", "Male"]
RelationshipWithHead = Literal[
    "Head of Household", "Spouse", "Child", "Parent", "Other relative", "Other non-relatives"
]
MaritalStatus = Literal[
    "Married/Living together", "Single/Never Married", "Widowed", "Divorced/Seperated", "Dont know"
]
EducationLevel = Literal[
    "No formal education",
    "Primary education",
    "Secondary education",
    "Tertiary education",
    "Vocational/Specialised training",
    "Other/Dont know/RTA",
]
JobType = Literal[
    "Farming and Fishing",
    "Self employed",
    "Formally employed Government",
    "Formally employed Private",
    "Informally employed",
    "Remittance Dependent",
    "Government Dependent",
    "Other Income",
    "No Income",
    "Dont Know/Refuse to answer",
]

# strict=True: reject 3.5, "3", true, NaN and Infinity outright instead of
# letting lax-mode coercion quietly turn them into integers. (Python's json
# parser accepts the non-standard NaN/Infinity tokens, so this matters.)
HouseholdSize = Annotated[
    int,
    Field(
        strict=True,
        ge=1,
        # Training max is 21; 30 leaves headroom for genuine extended
        # households, which EDA 1.6 found are real. Values above the training
        # max are still served but flagged as out-of-distribution.
        le=30,
        description="Number of people in the household (training range 1-21).",
    ),
]
Age = Annotated[
    int,
    Field(
        strict=True,
        ge=16,  # the Findex-derived surveys only interview adults 16+
        le=100,
        description="Respondent age in years (training range 16-100).",
    ),
]


class SurveyRecord(BaseModel):
    """One survey respondent. Field names match the training data columns."""

    model_config = ConfigDict(
        extra="forbid",  # a misspelled field must fail, not be silently ignored
        frozen=True,
        json_schema_extra={
            "examples": [
                {
                    "country": "Uganda",
                    "location_type": "Rural",
                    "cellphone_access": "Yes",
                    "household_size": 5,
                    "age_of_respondent": 34,
                    "gender_of_respondent": "Female",
                    "relationship_with_head": "Spouse",
                    "marital_status": "Married/Living together",
                    "education_level": "Primary education",
                    "job_type": "Self employed",
                }
            ]
        },
    )

    country: Country
    location_type: LocationType
    cellphone_access: YesNo
    household_size: HouseholdSize
    age_of_respondent: Age
    gender_of_respondent: Gender
    relationship_with_head: RelationshipWithHead
    marital_status: MaritalStatus
    education_level: EducationLevel
    job_type: JobType


def categorical_literals() -> dict[str, set[str]]:
    """The allowed values of every enum field, for the startup cross-check."""
    literals: dict[str, set[str]] = {}
    for name, field in SurveyRecord.model_fields.items():
        if typing.get_origin(field.annotation) is Literal:
            literals[name] = set(typing.get_args(field.annotation))
    return literals


class PredictionResponse(BaseModel):
    # `model_version` is a domain field, not pydantic API; opt out of the `model_` namespace guard.
    model_config = ConfigDict(protected_namespaces=())

    request_id: str
    model_version: str
    probability_bank_account: float = Field(
        description="Model score for bank_account == 'Yes'. The model was trained "
        "with scale_pos_weight, so scores sit above the population base rate: "
        "read it as a ranking score against decision_threshold, not a calibrated probability."
    )
    predicted_bank_account: Literal["Yes", "No"]
    confidence: float = Field(
        description="Probability the model assigns to the predicted class "
        "(probability if predicted 'Yes', else 1 - probability)."
    )
    decision_threshold: float
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal notes, e.g. an input outside the training range.",
    )


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: Literal["ok", "unavailable"]
    model_version: str | None = None
    detail: str | None = None


class ErrorResponse(BaseModel):
    error: str
    detail: object
    request_id: str | None = None
