"""readsync: offline-first, secure toolkit for synchronised reading experiments.

readsync presents reading materials, records gaze from a research-grade
eye-tracker, marks events for synchronisation with EEG, and stores everything
locally under encryption. It extends PsychoPy and the established analysis
tools; it does not replace them. See ARCHITECTURE.md for the design and
SECURITY.md for the data-protection guarantees.
"""

from __future__ import annotations

from .export import events_to_csv, log_to_csv
from .markers import EyeLinkMarkerSink, LSLMarkerSink, Marker, MarkerSink, NullMarkerSink
from .presenters import PsychoPyPresenter
from .quality import (
    PassageQuality,
    QualityReport,
    log_quality,
    quality_to_json,
    session_quality,
)
from .security import (
    DecryptionError,
    NetworkGuard,
    OfflineViolation,
    decrypt,
    encrypt,
    new_data_key,
    pseudonymise,
)
from .session import (
    HeadlessPresenter,
    Presenter,
    QuestionResponse,
    ReadingSession,
    SessionResult,
)
from .stimuli import Question, StimulusItem, StimulusSet, load_passages, load_stimulus_set
from .storage import EventLog, IntegrityError
from .text import (
    FixedWidthLayout,
    InterestArea,
    Passage,
    Region,
    Word,
    interest_areas,
    locate,
    region_at,
    tokenise,
)
from .trackers import EyeLinkTracker, GazeSample, NullTracker, Tracker
from .webcam import WebcamTracker

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # security
    "pseudonymise",
    "new_data_key",
    "encrypt",
    "decrypt",
    "DecryptionError",
    "NetworkGuard",
    "OfflineViolation",
    # storage and export
    "EventLog",
    "IntegrityError",
    "events_to_csv",
    "log_to_csv",
    # stimuli
    "Passage",
    "Word",
    "InterestArea",
    "FixedWidthLayout",
    "Region",
    "region_at",
    "tokenise",
    "interest_areas",
    "locate",
    "load_passages",
    "load_stimulus_set",
    "Question",
    "StimulusItem",
    "StimulusSet",
    # trackers and markers
    "Tracker",
    "NullTracker",
    "EyeLinkTracker",
    "GazeSample",
    "Marker",
    "MarkerSink",
    "NullMarkerSink",
    "LSLMarkerSink",
    "EyeLinkMarkerSink",
    "WebcamTracker",
    # session
    "ReadingSession",
    "SessionResult",
    "Presenter",
    "HeadlessPresenter",
    "PsychoPyPresenter",
    "QuestionResponse",
    # quality
    "PassageQuality",
    "QualityReport",
    "session_quality",
    "quality_to_json",
    "log_quality",
]
