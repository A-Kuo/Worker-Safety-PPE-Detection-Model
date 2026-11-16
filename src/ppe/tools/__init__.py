"""Model build tools shipped in the package (``ppe export`` / ``ppe quantize``).

They live here, not only under ``scripts/``, because ``scripts/`` is not part of the wheel. Heavy
dependencies are imported lazily and reported with the extra that provides them.
"""
