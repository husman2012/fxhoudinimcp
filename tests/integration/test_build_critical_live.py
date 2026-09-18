"""Success-path checks for the tools an agent-driven Houdini build depends on.

The rest of the suite exercises most commands, but several build-critical
ones only in smoke mode (a clean error also passes) or on trivial scenes.
These tests drive them the way a sparse-pyro build does -- node cards,
atomic builds with DOP wrangles, VEX validation, stepping, temporal
assertions, caching, HDA creation, a PDG cook, and a Karma render read back
from disk -- and assert on what Houdini actually produced.

Regression coverage for defects found running the suite on Houdini 22:
  * get_node_card returned no connector labels (claimed in its docs);
  * validate_vex / set_wrangle_code called broken Gas Field Wrangle VEX valid,
    and verify_network called a DOP net with uncompilable VEX healthy;
  * write_cache ignored frame_range on File Cache 2.0 (f1/f2 are
    $FSTART/$FEND expressions) and reported success when nothing was written;
  * build_network could not set a constant over a default expression;
  * setup_render's resolution did not reach Karma renders;
  * assign_material's SOP-level assignment was ignored by Karma renders;
  * render_lint_settings rejected the /out Karma ROP that setup_render creates;
  * cook_top_node reported success when work items failed or wrote nothing.
"""

from __future__ import annotations

# Built-in
import os

# Third-party
import hou
import pytest

pytestmark = pytest.mark.integration

GOOD_GAS_VEX = """\
// fuel burns with an Arrhenius-like rate; heat and expansion follow it
float T = f@temperature;
float Y = f@fuel;
float rate = 5.0 * Y * exp(-15000.0 / max(T, 1.0));
f@fuel = max(Y - rate * @TimeInc, 0.0);
f@temperature = T + 2000.0 * rate * @TimeInc;
f@divergence += 0.1 * rate;
"""
SYNTAX_ERROR_VEX = "f@temperature = max(f@temperature, 0.0) +;"
UNDEFINED_FN_VEX = "f@temperature = no_such_function_xyz(f@density);"
WARNING_ONLY_VEX = "int i = 1.5; f@temperature = i;"  # implicit float->int cast warning
_HDR = "Errors or warnings encountered during VEX compile:\n"


def _menu_set(parm: hou.Parm, label_part: str) -> None:
    labels = [label.lower() for label in parm.parmTemplate().menuLabels()]
    for index, label in enumerate(labels):
        if label_part.lower() in label:
            parm.set(index)
            return
    raise AssertionError(f"{parm.path()}: no menu label containing {label_part!r} in {labels}")


def _license_ext(kind: str) -> str:
    """hip/hda extension this license writes (Indie: .hiplc/.hdalc, Apprentice: nc)."""
    suffix = {
        hou.licenseCategoryType.Indie: "lc",
        hou.licenseCategoryType.Apprentice: "nc",
    }.get(hou.licenseCategory(), "")
    return kind + suffix


def _no_scratch_left() -> None:
    leftovers = [
        n.path() for n in hou.node("/obj").children() if n.name().startswith("__fxh_")
    ]
    assert not leftovers, f"scratch nodes left behind: {leftovers}"


###### get_node_card: connector labels


class TestNodeCardConnectors:
    @pytest.mark.parametrize(
        "node_type,context,container",
        [
            ("pyrosolver_sparse", "Dop", "dopnet"),
            ("gasfieldwrangle", "Dop", "dopnet"),
            ("copytopoints", "Sop", "geo"),
            ("volumerasterizeattributes", "Sop", "geo"),
        ],
    )
    def test_card_reports_the_real_connector_labels(self, call, node_type, context, container):
        card = call("graph.get_node_card", node_type=node_type, context=context)
        # Ground truth: the labels of a real instance in this Houdini.
        net = hou.node("/obj").createNode(container, "truth")
        instance = net.createNode(node_type)
        assert card["inputs"] == list(instance.inputLabels())[: len(card["inputs"])]
        assert len(card["inputs"]) == min(len(instance.inputLabels()), 16)
        assert card["outputs"] == list(instance.outputLabels())[: len(card["outputs"])]
        assert card["inputs"], f"no connector labels for {node_type}"
        _no_scratch_left()

    def test_card_probe_leaves_the_scene_untouched(self, call):
        before = sorted(n.path() for n in hou.node("/obj").allSubChildren())
        call("graph.get_node_card", node_type="pyrosolver_sparse", context="Dop")
        call("graph.get_node_card", node_type="geo", context="Object")
        after = sorted(n.path() for n in hou.node("/obj").allSubChildren())
        assert before == after


###### VEX validation of DOP wrangles


@pytest.fixture
def gas_wrangle(call) -> str:
    call("graph.build_network", parent_path="/obj", nodes=[{"type": "dopnet", "name": "sim"}])
    call(
        "graph.build_network",
        parent_path="/obj/sim",
        nodes=[{"type": "gasfieldwrangle", "name": "chem", "parms": {"snippet": GOOD_GAS_VEX}}],
    )
    return "/obj/sim/chem"


class TestDopWrangleValidation:
    def test_valid_combustion_vex_is_valid(self, call, gas_wrangle):
        result = call("vex.validate_vex", node_path=gas_wrangle)
        assert result["is_valid"] is True, result
        assert result["compile_checked"] is True
        assert result["compile_method"] == "scratch_dop_solve"
        _no_scratch_left()

    @pytest.mark.parametrize(
        "code,needle",
        [(SYNTAX_ERROR_VEX, "syntax error"), (UNDEFINED_FN_VEX, "undefined function")],
    )
    def test_broken_vex_is_invalid(self, call, gas_wrangle, code, needle):
        hou.setFrame(7)
        written = call("vex.set_wrangle_code", node_path=gas_wrangle, vex_code=code)
        assert written["vex_valid"] is False, written
        result = call("vex.validate_vex", node_path=gas_wrangle)
        assert result["is_valid"] is False, result
        assert any(needle in e.lower() for e in result["errors"]), result["errors"]
        # The check must not move the playbar or leave nodes behind.
        assert hou.frame() == 7
        _no_scratch_left()

    def test_compile_warning_alone_is_not_an_error(self, call, gas_wrangle):
        # Code that compiles with at most a warning stays valid. Houdini does
        # not reliably re-report compile warnings (a recompile of identical
        # code reports none), so the classification of the warning text is
        # checked on real H22 messages below instead.
        written = call("vex.set_wrangle_code", node_path=gas_wrangle, vex_code=WARNING_ONLY_VEX)
        result = call("vex.validate_vex", node_path=gas_wrangle)
        assert written["vex_valid"] is True, written
        assert result["is_valid"] is True, result

    @pytest.mark.parametrize(
        "message,is_error",
        [
            # Messages Houdini 22.0.429 attached to Gas Field / POP Wrangles.
            ("Error in VOP 'snippet1'.", True),
            (_HDR + "snippet1: Syntax error, unexpected ';'.\t (1,60).", True),
            (_HDR + "snippet1: Call to undefined function 'undefined_fn_xyz'.\t (1,26:41).", True),
            ("Call to undefined function 'undefined_fn_xyz'.\t (1,17:32)", True),
            ("smokeobject1 - /obj/x/chem/gasfieldvop1: snippet1: Syntax error, unexpected ';'."
             "\nFailed to resolve VEX code op:/obj/x/chem/gasfieldvop1", True),
            (_HDR + "snippet1: Implicit cast from float to int. Use explicit cast instead.\t (1,9).", False),
            ("Implicit cast from float to int. Use explicit cast instead.\t (1,5)", False),
            ("Could not create directory 'Q:/no/such/drive'.", False),
        ],
    )
    def test_compile_message_classification(self, message, is_error):
        from fxhoudinimcp_server.handlers.vex_handlers import _is_vex_compile_error

        assert _is_vex_compile_error(message) is is_error

    def test_broken_pop_wrangle_is_invalid(self, call):
        call("graph.build_network", parent_path="/obj", nodes=[{"type": "dopnet", "name": "parts"}])
        call(
            "graph.build_network",
            parent_path="/obj/parts",
            nodes=[{"type": "popwrangle", "name": "w", "parms": {"snippet": "v@v = no_such_fn(1);"}}],
        )
        result = call("vex.validate_vex", node_path="/obj/parts/w")
        assert result["is_valid"] is False, result

    def test_sop_wrangle_path_unchanged(self, call):
        geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="g")["node_path"]
        created = call("vex.create_wrangle", parent_path=geo, vex_code="@P.y += 1;")
        assert created["vex_valid"] is True
        result = call("vex.set_wrangle_code", node_path=created["node_path"], vex_code=SYNTAX_ERROR_VEX.replace("f@temperature", "f@x"))
        assert result["vex_valid"] is False
        assert call("vex.validate_vex", node_path=created["node_path"])["compile_method"] == "cook"


###### An end-to-end sparse pyro build through the build-critical tools


def _build_source(call, voxel: float) -> str:
    call("graph.build_network", parent_path="/obj", nodes=[{"type": "geo", "name": "src"}])
    call(
        "graph.build_network",
        parent_path="/obj/src",
        nodes=[
            {"type": "sphere", "name": "shape", "parms": {"type": "poly", "rad": [0.3, 0.3, 0.3], "ty": 0.3, "freq": 8}},
            {"type": "pyrosource", "name": "psrc", "inputs": ["shape"]},
            {"type": "volumerasterizeattributes", "name": "rast", "inputs": ["psrc"],
             "parms": {"attributes": "density temperature", "voxelsize": voxel}},
            {"type": "null", "name": "OUT", "inputs": ["rast"], "flags": {"display": True}},
        ],
    )
    psrc = hou.node("/obj/src/psrc")
    _menu_set(psrc.parm("mode"), "volume scatter")
    psrc.parm("particlesep").set(voxel)
    psrc.parm("attributes").set(2)
    _menu_set(psrc.parm("attribute1"), "density")
    _menu_set(psrc.parm("attribute2"), "temperature")
    report = call("graph.verify_network", parent_path="/obj/src")
    assert report["healthy"], report["error_nodes"]
    assert report["geometry"]["prims"] >= 2, f"source rasterized nothing: {report['geometry']}"
    return "/obj/src/OUT"


def _build_pyro_dop(call, source_sop: str, vex: str) -> str:
    card = call("graph.get_node_card", node_type="pyrosolver_sparse", context="Dop")
    labels = [label.lower() for label in card["inputs"]]
    sourcing = next(i for i, label in enumerate(labels) if "sourc" in label)
    forces = next(i for i, label in enumerate(labels) if "force" in label)
    call("graph.build_network", parent_path="/obj", nodes=[{"type": "dopnet", "name": "sim"}])
    built = call(
        "graph.build_network",
        parent_path="/obj/sim",
        nodes=[
            {"type": "smokeobject_sparse", "name": "pyro", "parms": {"divsize": 0.1}},
            {"type": "volumesource", "name": "source"},
            {"type": "gasfieldwrangle", "name": "chem", "parms": {"snippet": vex}},
            {"type": "pyrosolver_sparse", "name": "solver", "inputs": [
                {"index": 0, "source": "pyro"},
                {"index": sourcing, "source": "source"},
                {"index": forces, "source": "chem"},
            ]},
            {"type": "output", "name": "out", "inputs": ["solver"], "flags": {"display": True}},
        ],
    )
    assert built["valid"], built
    vs = hou.node("/obj/sim/source")
    vs.parm("initialize").set("sourcing")
    vs.parm("initialize").pressButton()
    vs.parm("soppath").set(source_sop)
    solver = hou.node("/obj/sim/solver")
    assert solver.inputs()[forces].name() == "chem"
    return "/obj/sim"


def _field_importer(call, dopnet: str, field: str) -> str:
    call("graph.build_network", parent_path="/obj", nodes=[{"type": "geo", "name": "fields"}])
    call(
        "graph.build_network",
        parent_path="/obj/fields",
        nodes=[{"type": "dopimportfield", "name": "imp",
                "parms": {"doppath": dopnet, "defobj": "pyro", "fields": 1},
                "flags": {"display": True}}],
    )
    # fieldname1 only exists once the multiparm has an entry.
    hou.node("/obj/fields/imp").parm("fieldname1").set(field)
    return "/obj/fields/imp"


class TestSparsePyroBuild:
    def test_step_inspect_and_assert(self, call):
        dopnet = _build_pyro_dop(call, _build_source(call, 0.1), GOOD_GAS_VEX)
        assert call("graph.verify_network", parent_path=dopnet)["healthy"]
        assert call("vex.validate_vex", node_path=f"{dopnet}/chem")["is_valid"]

        hou.setFrame(1)
        stepped = call("dops.step_simulation", node_path=dopnet, steps=5)
        assert stepped["end_frame"] == 6
        info = call("dops.get_simulation_info", node_path=dopnet)
        assert info["object_count"] >= 1
        assert info["simulation_time"] > 0
        assert info["memory_usage_bytes"] > 0

        importer = _field_importer(call, dopnet, "density")
        passing = call(
            "assert_simulation",
            network=importer,
            frame_range=[1, 6],
            assertions=[{"metric": "bbox_over_time", "max": 50.0},
                        {"metric": "point_count", "min": 1}],
        )
        assert passing["pass"] is True, passing
        series = passing["results"][0]["series"]
        assert series[-1][1] > 0.0, series
        failing = call(
            "assert_simulation",
            network=importer,
            frame_range=[1, 6],
            assertions=[{"metric": "bbox_over_time", "max": 1e-6}],
        )
        assert failing["pass"] is False, failing

    def test_broken_forces_wrangle_is_flagged_after_stepping(self, call):
        dopnet = _build_pyro_dop(call, _build_source(call, 0.1), SYNTAX_ERROR_VEX)
        hou.setFrame(1)
        call("dops.step_simulation", node_path=dopnet, steps=2)
        report = call("graph.verify_network", parent_path=dopnet)
        assert report["healthy"] is False, report
        assert f"{dopnet}/chem" in report["error_nodes"]
        # Fixing the code clears the stale compile messages of that solve.
        fixed = call("vex.set_wrangle_code", node_path=f"{dopnet}/chem", vex_code=GOOD_GAS_VEX)
        assert fixed["vex_valid"] is True, fixed
        assert call("graph.verify_network", parent_path=dopnet)["healthy"] is True

    def test_cache_the_simulated_fields(self, call, tmp_path):
        dopnet = _build_pyro_dop(call, _build_source(call, 0.1), GOOD_GAS_VEX)
        importer = _field_importer(call, dopnet, "density")
        pattern = str(tmp_path / "density.$F4.bgeo.sc").replace("\\", "/")
        call(
            "graph.build_network",
            parent_path="/obj/fields",
            nodes=[{"type": "filecache", "name": "cache", "inputs": [importer],
                    "parms": {"filemethod": 1, "file": pattern}}],
        )
        result = call("cache.write_cache", node_path="/obj/fields/cache", frame_range=[1, 4])
        assert result["files_verified"] == 4, result
        written = sorted(p.name for p in tmp_path.glob("density.*.bgeo.sc"))
        assert written == [f"density.{f:04d}.bgeo.sc" for f in (1, 2, 3, 4)]
        last = hou.Geometry()
        last.loadFromFile(str(tmp_path / "density.0004.bgeo.sc"))
        assert last.prims(), "cached frame has no volume primitives"


###### write_cache / build_network expression overrides


@pytest.fixture
def box_cache(call, tmp_path):
    geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="cached")["node_path"]
    call(
        "graph.build_network",
        parent_path=geo,
        nodes=[{"type": "box", "name": "b"},
               {"type": "filecache", "name": "cache", "inputs": ["b"],
                "parms": {"filemethod": 1, "file": str(tmp_path / "box.$F4.bgeo.sc").replace("\\", "/")}}],
    )
    return f"{geo}/cache"


class TestWriteCache:
    def test_frame_range_is_honoured(self, call, box_cache, tmp_path):
        result = call("cache.write_cache", node_path=box_cache, frame_range=[2, 4])
        assert result["frames_requested"] == 3
        written = sorted(p.name for p in tmp_path.glob("box.*.bgeo.sc"))
        assert written == ["box.0002.bgeo.sc", "box.0003.bgeo.sc", "box.0004.bgeo.sc"]

    def test_constructed_paths_are_verified(self, call, box_cache, tmp_path):
        # Default File Cache 2.0 mode builds the path from basedir/basename/
        # version; the Explicit-mode "file" parm is then unused.
        node = hou.node(box_cache)
        node.parm("filemethod").set(0)
        node.parm("basedir").set(str(tmp_path / "constructed").replace("\\", "/"))
        result = call("cache.write_cache", node_path=box_cache, frame_range=[1, 2])
        assert result["files_verified"] == 2, result
        assert len(list((tmp_path / "constructed").rglob("*.bgeo.sc"))) == 2

    def test_failed_write_is_an_error(self, call, box_cache, tmp_path):
        blocker = tmp_path / "not_a_dir"
        blocker.write_text("a file where the cache directory should be")
        hou.node(box_cache).parm("file").set(str(blocker / "box.$F4.bgeo.sc").replace("\\", "/"))
        error = call("cache.write_cache", node_path=box_cache, frame_range=[1, 2], expect_error=True)
        assert "write_cache failed" in error["message"], error

    def test_rop_geometry_range(self, call, tmp_path):
        geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="g")["node_path"]
        box = call("nodes.create_node", parent_path=geo, node_type="box")["node_path"]
        call(
            "graph.build_network",
            parent_path="/out",
            nodes=[{"type": "geometry", "name": "rop",
                    "parms": {"soppath": box, "sopoutput": str(tmp_path / "rop.$F4.bgeo.sc").replace("\\", "/")}}],
        )
        result = call("cache.write_cache", node_path="/out/rop", frame_range=[3, 5])
        assert result["files_verified"] == 3
        assert sorted(p.name for p in tmp_path.glob("rop.*.bgeo.sc")) == [
            "rop.0003.bgeo.sc", "rop.0004.bgeo.sc", "rop.0005.bgeo.sc"]

    def test_build_network_constant_replaces_expression(self, call):
        geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="g")["node_path"]
        call(
            "graph.build_network",
            parent_path=geo,
            nodes=[{"type": "filecache", "name": "fc", "parms": {"f1": 5, "f2": 9}}],
        )
        node = hou.node(f"{geo}/fc")
        assert (node.parm("f1").eval(), node.parm("f2").eval()) == (5, 9)
        assert not node.parm("f2").keyframes()


###### HDA creation (license-appropriate extension)


class TestCreateHda:
    def test_subnet_with_a_dop_network_becomes_an_asset(self, call, tmp_path):
        ext = _license_ext("hda")
        sub = hou.node("/obj").createNode("subnet", "fx_asset")
        sub.createNode("dopnet", "sim").createNode("smokeobject_sparse")
        path = str(tmp_path / f"s10_fx_asset.{ext}").replace("\\", "/")
        type_name = "fxhtest::fx_asset"
        created = call("hda.create_hda", node_path=sub.path(), hda_file=path,
                       type_name=type_name, label="FX Asset")
        try:
            assert os.path.isfile(path)
            node = hou.node(created["node_path"])
            assert node.type().name() == type_name
            second = hou.node("/obj").createNode(type_name, "fx_asset2")
            assert second.node("sim/smokeobject_sparse1") is not None
        finally:
            hou.hda.uninstallFile(path)

    def test_dop_subnet_becomes_an_asset(self, call, tmp_path):
        net = hou.node("/obj").createNode("dopnet", "sim")
        sub = net.createNode("subnet", "fx")
        sub.createNode("gasfieldwrangle", "w").parm("snippet").set("f@density *= 0.99;")
        path = str(tmp_path / f"s10_dop_fx.{_license_ext('hda')}").replace("\\", "/")
        call("hda.create_hda", node_path=sub.path(), hda_file=path,
             type_name="fxhtest::dop_fx", label="DOP FX")
        try:
            second = net.createNode("fxhtest::dop_fx", "fx2")
            # H22 creates a DOP asset's contents lazily: children() is empty
            # on a fresh instance, but the contents are there when asked for.
            assert second.node("w") is not None
            assert second.node("w").parm("snippet").eval() == "f@density *= 0.99;"
        finally:
            hou.hda.uninstallFile(path)


###### PDG cook


class TestTopCook:
    def test_python_work_items_cook(self, call):
        call("graph.build_network", parent_path="/obj", nodes=[{"type": "topnet", "name": "tops"}])
        call(
            "graph.build_network",
            parent_path="/obj/tops",
            nodes=[{"type": "genericgenerator", "name": "gen", "parms": {"itemcount": 4}},
                   {"type": "pythonscript", "name": "py", "inputs": ["gen"],
                    "parms": {"script": "work_item.setIntAttrib('sq', work_item.index ** 2)"}}],
        )
        cooked = call("tops.cook_top_node", node_path="/obj/tops/py", block=True)
        assert cooked["work_item_count"] == 4 and not cooked["errors"], cooked
        assert cooked["state_counts"] == {"cooked_success": 4}, cooked
        states = call("tops.get_work_item_states", node_path="/obj/tops/py")
        assert states["state_counts"] == {"cooked_success": 4}, states

    def test_failing_script_is_an_error(self, call):
        call("graph.build_network", parent_path="/obj", nodes=[{"type": "topnet", "name": "tops"}])
        call(
            "graph.build_network",
            parent_path="/obj/tops",
            nodes=[{"type": "genericgenerator", "name": "gen", "parms": {"itemcount": 2}},
                   {"type": "pythonscript", "name": "py", "inputs": ["gen"],
                    "parms": {"script": "raise RuntimeError('deliberate failure')"}}],
        )
        error = call("tops.cook_top_node", node_path="/obj/tops/py", block=True, expect_error=True)
        assert "deliberate failure" in error["message"], error

    def _rop_fetch(self, call, out_pattern: str, cook_type: int) -> str:
        geo = call("nodes.create_node", parent_path="/obj", node_type="geo", name="g")["node_path"]
        box = call("nodes.create_node", parent_path=geo, node_type="box")["node_path"]
        call("graph.build_network", parent_path="/out", nodes=[
            {"type": "geometry", "name": "rop", "parms": {"soppath": box, "sopoutput": out_pattern}}])
        call("graph.build_network", parent_path="/obj", nodes=[{"type": "topnet", "name": "tops"}])
        call("graph.build_network", parent_path="/obj/tops", nodes=[
            {"type": "ropfetch", "name": "fetch", "parms": {
                "roppath": "/out/rop", "framegeneration": 1, "range1": 6, "range2": 7,
                "pdg_cooktype": cook_type}}])
        return "/obj/tops/fetch"

    def test_rop_fetch_of_saved_scene_writes_its_frames(self, call, tmp_path):
        fetch = self._rop_fetch(call, str(tmp_path / "f.$F4.bgeo.sc").replace("\\", "/"), cook_type=1)
        hou.hipFile.save(str(tmp_path / f"scene.{_license_ext('hip')}").replace("\\", "/"))
        cooked = call("tops.cook_top_node", node_path=fetch, block=True)
        assert cooked["state_counts"] == {"cooked_success": 2}, cooked
        assert sorted(p.name for p in tmp_path.glob("f.*.bgeo.sc")) == ["f.0006.bgeo.sc", "f.0007.bgeo.sc"]

    def test_rop_fetch_that_writes_nothing_is_an_error(self, call, tmp_path):
        # In-process ROP Fetch of an UNSAVED scene: PDG marks the items
        # cooked but the files never appear (H22.0.429).
        fetch = self._rop_fetch(call, str(tmp_path / "f.$F4.bgeo.sc").replace("\\", "/"), cook_type=0)
        error = call("tops.cook_top_node", node_path=fetch, block=True, expect_error=True)
        assert "missing" in error["message"] or "failed" in error["message"], error
        assert not list(tmp_path.glob("f.*.bgeo.sc"))


###### Material -> Karma render -> read back from disk


def _karma_scene(call, tmp_path, color, name):
    material = call("workflow.create_material", name=name, mat_type="principled", base_color=color)
    call("workflow.assign_material", geo_path="/obj/ball", material_path=material["material_path"])
    out = str(tmp_path / f"{name}.$F4.exr").replace("\\", "/")
    render = call("workflow.setup_render", renderer="karma", output_path=out,
                  resolution=[64, 48], samples=4, name=f"karma_{name}")
    hou.node(render["camera_path"]).parmTuple("t").set((0, 0, 5))
    rendered = call("rendering.start_render", node_path=render["rop_path"], frame_range=[1, 1])
    assert rendered["success"], rendered
    # start_render waits for the render (soho_foreground) only for its own
    # call; the ROP keeps its setting. (In hython render() blocks anyway;
    # the graphical-session behaviour is covered by the S10 GUI replay.)
    assert hou.node(render["rop_path"]).parm("soho_foreground").eval() == 0
    exr = str(tmp_path / f"{name}.0001.exr").replace("\\", "/")
    assert os.path.isfile(exr), f"no image written: {os.listdir(tmp_path)}"
    return render["rop_path"], exr


class TestKarmaRenderReadback:
    def test_render_is_read_back_and_compared(self, call, tmp_path):
        ball = hou.node("/obj").createNode("geo", "ball")
        ball.createNode("sphere")
        _, red_exr = _karma_scene(call, tmp_path, [0.9, 0.05, 0.05], "red")
        # A second Material SOP downstream re-points the sphere at blue.
        rop, blue_exr = _karma_scene(call, tmp_path, [0.05, 0.05, 0.9], "blue")

        manifest = call("render_parse_exr", exr_path=red_exr, subimage=None)
        assert (manifest["xres"], manifest["yres"]) == (64, 48), manifest

        red = call("render_read_pixels", source=red_exr, plane="C", mode="summary",
                   roi=None, max_pixels=4096, downsample=1, page=0, page_size=1024)
        blue = call("render_read_pixels", source=blue_exr, plane="C", mode="summary",
                    roi=None, max_pixels=4096, downsample=1, page=0, page_size=1024)
        red_mean, blue_mean = red["stats"]["mean"], blue["stats"]["mean"]
        assert red_mean[0] > 2 * red_mean[2], red_mean
        assert blue_mean[2] > 2 * blue_mean[0], blue_mean

        compared = call("render_compare", a=red_exr, b=blue_exr, planes=None, metric="stats")
        plane = next(p for p in compared["per_plane"] if p["plane"] == "C")
        # mean_delta is mean(a - b): red minus blue.
        assert plane["mean_delta"][0] > 0 > plane["mean_delta"][2], plane
        assert compared["verdict"] != "no change", compared

        lint = call("render_lint_settings", render_node=rop, preset="nuke_safe")
        assert "results" in lint and lint.get("ok") is not False, lint
        assert lint["stage_node"].startswith(rop + "/"), lint
