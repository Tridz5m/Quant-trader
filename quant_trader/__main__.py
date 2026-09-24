import sys

from ._deps import check_dependencies

check_dependencies()

from .cli import main  # noqa: E402

sys.exit(main())
