"""Pure-logic pytest tests for cooked_from_errors / normalize_plane_dtype
(PP12-113 PR-4 -- appended to cop_onnx_model.py).

Unit: pp12-113d
testVerificationSurface: pytest-model
planSha: 92ce0bfd3ac81683321af721d8bff6bd50e7c67010fdfca5e82f53f97845adc9

A NEW test file (not an edit to the existing test_cop_onnx_model.py) per
the append-only / test-files-only discipline -- PR-1/PR-2/PR-3 test
sections stay byte-unchanged.

These tests are written BEFORE the implementation (red phase). They will
fail with ImportError until hou-dev appends cooked_from_errors and
normalize_plane_dtype to fxhoudinimcp/cop_onnx_model.py.

Contract (plan pp12-113d lockedFieldContract, "pure helpers:
cooked_from_errors(errors) + normalize_plane_dtype(storage_type)"):

    cooked_from_errors(errors: list) -> bool
        return not errors  -- pins the FR-5 no-silent-success invariant
        (cooked iff zero cook errors).

    normalize_plane_dtype(storage_type) -> str
        return str(storage_type).rsplit('.', 1)[-1].lower()  -- maps a
        hou.imageLayerStorageType enum (e.g. imageLayerStorageType.Float32)
        or its str to a plain dtype token 'float32'; accepts a str OR the
        enum (str() first), so it is pure + off-DCC testable with
        plain-string inputs.

Both are pure -- no hou/Qt/pxr/onnx/numpy imports anywhere in this file or
in the two functions under test (CL-015).
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# module import + callable (PRIMARY RED GATE)
# ---------------------------------------------------------------------------

class TestModuleImport:
    def test_module_importable(self):
        """cop_onnx_model must remain importable (PR-1/PR-2/PR-3 baseline)."""
        import fxhoudinimcp.cop_onnx_model  # noqa: F401

    def test_existing_symbols_unaffected(self):
        """PR-1/PR-2/PR-3 symbols must still be importable -- the new
        helpers must not clobber their existing siblings (append-only
        contract)."""
        from fxhoudinimcp.cop_onnx_model import (  # noqa: F401
            OnnxContract,
            TensorSpec,
            choose_provider,
            clamp_readback,
            contract_from_setup_shapes,
            count_nan_inf,
            guess_layout,
            paginate,
        )
        assert callable(choose_provider)
        assert callable(guess_layout)

    def test_cooked_from_errors_callable_on_module(self):
        """cooked_from_errors must be a callable. FAILS RED until hou-dev
        adds it."""
        from fxhoudinimcp.cop_onnx_model import cooked_from_errors  # noqa: F401
        assert callable(cooked_from_errors), (
            "cooked_from_errors must be a callable on fxhoudinimcp.cop_onnx_model."
        )

    def test_normalize_plane_dtype_callable_on_module(self):
        """normalize_plane_dtype must be a callable. FAILS RED until
        hou-dev adds it."""
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype  # noqa: F401
        assert callable(normalize_plane_dtype), (
            "normalize_plane_dtype must be a callable on fxhoudinimcp.cop_onnx_model."
        )


# ---------------------------------------------------------------------------
# cooked_from_errors — the FR-5 no-silent-success predicate
# ---------------------------------------------------------------------------

class TestCookedFromErrors:
    """cooked_from_errors(errors: list) -> bool. Pins the FR-5
    no-silent-success invariant: cooked iff the errors list is empty. A
    raised-but-empty-errors cook is folded to a non-empty errors list
    BEFORE this predicate runs (handler-side, per the LOCKED cook
    algorithm order) -- this pure predicate itself only ever sees the
    already-folded list."""

    def test_empty_list_is_cooked_true(self):
        from fxhoudinimcp.cop_onnx_model import cooked_from_errors

        assert cooked_from_errors([]) is True

    def test_non_empty_list_is_cooked_false(self):
        from fxhoudinimcp.cop_onnx_model import cooked_from_errors

        assert cooked_from_errors(["src is missing"]) is False

    def test_multiple_errors_is_cooked_false(self):
        from fxhoudinimcp.cop_onnx_model import cooked_from_errors

        assert cooked_from_errors([
            "The input data size from node input (3145728) does not match "
            "the expected size of the model input (12288).",
            "Error occurred extracting model input 1.",
        ]) is False

    def test_returns_a_bool(self):
        from fxhoudinimcp.cop_onnx_model import cooked_from_errors

        assert isinstance(cooked_from_errors([]), bool)
        assert isinstance(cooked_from_errors(["x"]), bool)


# ---------------------------------------------------------------------------
# normalize_plane_dtype — hou.imageLayerStorageType enum/str -> plain token
# ---------------------------------------------------------------------------

class TestNormalizePlaneDtype:
    """normalize_plane_dtype(storage_type) -> str. Maps a
    hou.imageLayerStorageType enum (grounded live: str() ==
    'imageLayerStorageType.Float32') or its already-stringified form to a
    plain lowercase dtype token."""

    def test_full_enum_str_form(self):
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype

        assert normalize_plane_dtype("imageLayerStorageType.Float32") == "float32"

    def test_object_whose_str_is_the_enum_form(self):
        """Accepts an object (e.g. the real enum member) whose str() is the
        enum-qualified form -- str() is applied first, per the locked
        contract."""
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype

        class _FakeStorageType:
            def __str__(self):
                return "imageLayerStorageType.Float32"

        assert normalize_plane_dtype(_FakeStorageType()) == "float32"

    def test_plain_unqualified_string(self):
        """A plain 'Float32' (no dotted qualifier) also normalizes to
        'float32' -- rsplit('.', 1)[-1] on a string with no '.' returns the
        whole string unchanged, then .lower()."""
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype

        assert normalize_plane_dtype("Float32") == "float32"

    def test_other_dtype_tokens(self):
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype

        assert normalize_plane_dtype("imageLayerStorageType.Int8") == "int8"
        assert normalize_plane_dtype("imageLayerStorageType.UInt16") == "uint16"

    def test_returns_a_str(self):
        from fxhoudinimcp.cop_onnx_model import normalize_plane_dtype

        assert isinstance(normalize_plane_dtype("imageLayerStorageType.Float32"), str)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
