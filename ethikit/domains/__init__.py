"""Reference suites: curated public datasets per domain, run through the same ethics battery
to give a *prior* about which kinds of model tend to do well -- for teams with no data of
their own yet. See ethikit/domains/spec.py for what that does and does not mean."""
from .registry import DEFAULT_STATES, get_domains  # noqa: F401
from .runner import DomainResult, compare_domains, run_domain, shortlist  # noqa: F401
from .spec import (DatasetSpec, DatasetUnavailable, Domain, STRENGTH_TEXT,  # noqa: F401
                   cache_dir, load_spec)
