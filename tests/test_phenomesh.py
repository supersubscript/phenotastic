"""Tests for the PhenoMesh class."""

import numpy as np
import pyvista as pv

from phenotastic import PhenoMesh


class TestPhenoMeshInheritance:
    """Tests for PhenoMesh inheritance from pv.PolyData."""

    def test_phenomesh_is_polydata(self) -> None:
        """PhenoMesh should be an instance of pv.PolyData."""
        mesh = PhenoMesh(pv.Sphere())
        assert isinstance(mesh, pv.PolyData)

    def test_phenomesh_is_phenomesh(self) -> None:
        """PhenoMesh should be an instance of PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        assert isinstance(mesh, PhenoMesh)

    def test_polydata_methods_available(self) -> None:
        """PolyData methods should be available on PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        # These are inherited from PolyData
        assert hasattr(mesh, "n_points")
        assert hasattr(mesh, "n_cells")
        assert hasattr(mesh, "points")
        assert hasattr(mesh, "faces")
        assert hasattr(mesh, "bounds")
        assert hasattr(mesh, "center")
        assert hasattr(mesh, "area")
        assert hasattr(mesh, "volume")

    def test_phenomesh_can_be_passed_to_polydata_functions(self) -> None:
        """PhenoMesh should be usable anywhere pv.PolyData is expected."""
        mesh = PhenoMesh(pv.Sphere())
        # pv.PolyData.merge expects PolyData arguments
        merged = mesh.merge(pv.Box())
        assert isinstance(merged, pv.PolyData)


class TestPhenoMeshCustomAttributes:
    """Tests for PhenoMesh custom attributes."""

    def test_contour_attribute(self) -> None:
        """PhenoMesh should store contour attribute."""
        contour = np.zeros((10, 10, 10), dtype=bool)
        mesh = PhenoMesh(pv.Sphere(), contour=contour)
        assert mesh.contour is not None
        np.testing.assert_array_equal(mesh.contour, contour)

    def test_resolution_attribute(self) -> None:
        """PhenoMesh should store resolution attribute."""
        resolution = [1.0, 2.0, 3.0]
        mesh = PhenoMesh(pv.Sphere(), resolution=resolution)
        assert mesh.resolution == resolution

    def test_copy_preserves_attributes(self) -> None:
        """Copy should preserve custom attributes."""
        contour = np.zeros((10, 10, 10), dtype=bool)
        resolution = [1.0, 2.0, 3.0]
        mesh = PhenoMesh(pv.Sphere(), contour=contour, resolution=resolution)

        copied = mesh.copy()

        assert isinstance(copied, PhenoMesh)
        np.testing.assert_array_equal(copied.contour, contour)
        assert copied.resolution == resolution


class TestPhenoMeshOperations:
    """Tests for PhenoMesh operations returning PhenoMesh."""

    def test_smooth_returns_phenomesh(self) -> None:
        """Smooth should return a PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        result = mesh.smooth(iterations=5)
        assert isinstance(result, PhenoMesh)

    def test_decimate_returns_phenomesh(self) -> None:
        """Decimate should return a PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        result = mesh.decimate(target_reduction=0.5)
        assert isinstance(result, PhenoMesh)

    def test_clean_returns_phenomesh(self) -> None:
        """Clean should return a PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        result = mesh.clean()
        assert isinstance(result, PhenoMesh)

    def test_extract_largest_returns_phenomesh(self) -> None:
        """Extract largest should return a PhenoMesh."""
        mesh = PhenoMesh(pv.Sphere())
        result = mesh.extract_largest()
        assert isinstance(result, PhenoMesh)

    def test_triangulate_returns_phenomesh(self) -> None:
        """Triangulate should return a PhenoMesh."""
        mesh = PhenoMesh(pv.Box())
        result = mesh.triangulate()
        assert isinstance(result, PhenoMesh)

    def test_smooth_preserves_attributes(self) -> None:
        """Operations should preserve custom attributes."""
        resolution = [1.0, 2.0, 3.0]
        mesh = PhenoMesh(pv.Sphere(), resolution=resolution)

        result = mesh.smooth(iterations=5)

        assert result.resolution == resolution


class TestPhenoMeshConversion:
    """Tests for PhenoMesh conversion methods."""

    def test_to_polydata_returns_polydata(self) -> None:
        """to_polydata should return a plain PolyData."""
        mesh = PhenoMesh(pv.Sphere())
        polydata = mesh.to_polydata()
        assert isinstance(polydata, pv.PolyData)
        # Should be a copy, not the same object
        assert polydata is not mesh

    def test_from_polydata_returns_phenomesh(self) -> None:
        """from_polydata should return a PhenoMesh."""
        polydata = pv.Sphere()
        mesh = PhenoMesh.from_polydata(polydata)
        assert isinstance(mesh, PhenoMesh)


class TestPhenoMeshBackwardsCompatibility:
    """Tests for backwards compatibility aliases."""

    def test_smoothen_alias(self) -> None:
        """smoothen should be an alias for smooth."""
        mesh = PhenoMesh(pv.Sphere())
        result = mesh.smoothen(iterations=5)
        assert isinstance(result, PhenoMesh)

    def test_boundary_points_alias(self) -> None:
        """boundary_points should be an alias for get_boundary_points."""
        mesh = PhenoMesh(pv.Sphere())
        # Sphere has no boundary, so this should return empty
        result = mesh.boundary_points()
        assert isinstance(result, np.ndarray)

    def test_filter_curvature_alias(self) -> None:
        """filter_curvature should be an alias for filter_by_curvature."""
        mesh = PhenoMesh(pv.Sphere())
        # Use wide threshold to keep vertices
        result = mesh.filter_curvature(curvature_threshold=5.0)
        assert isinstance(result, PhenoMesh)
