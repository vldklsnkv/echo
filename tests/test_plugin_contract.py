from __future__ import annotations

import json
import re
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOT = PLUGIN_ROOT / "skills" / "echo"


def _yaml_interface() -> dict[str, str]:
    lines = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8").splitlines()
    result: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r'  ([a-z_]+): "([^"]+)"', line)
        if match:
            result[match.group(1)] = match.group(2)
    return result


def test_plugin_manifest_has_only_implemented_components_and_coherent_metadata() -> None:
    manifest = json.loads((PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text())

    assert manifest["name"] == PLUGIN_ROOT.name == "echo"
    assert re.fullmatch(
        r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?", manifest["version"]
    )
    assert manifest["version"].partition("+")[0] == "0.1.0"
    assert manifest["author"]["name"].strip()
    assert manifest["skills"] == "./skills/"
    assert all(
        not Path(value).is_absolute() and value.startswith("./") for value in [manifest["skills"]]
    )
    assert not {"apps", "mcpServers", "hooks", "icon", "logo", "screenshots", "assets"} & set(
        manifest
    )
    assert "TODO" not in json.dumps(manifest).upper()

    interface = manifest["interface"]
    assert interface["displayName"] == "Echo"
    assert "GigaAM" in manifest["description"]
    assert interface["shortDescription"].rstrip(".") == (
        "Local Russian-first meeting transcription"
    )
    assert "word timestamps" in interface["longDescription"]
    assert "local diarization" in interface["longDescription"]
    assert interface["defaultPrompt"] == [
        "Use $echo:echo to transcribe this recording locally with speaker labels."
    ]
    assert interface["capabilities"] == []
    assert interface["composerIcon"] == "./assets/icon.png"
    assert interface["logo"] == "./assets/icon.png"
    assert (PLUGIN_ROOT / interface["composerIcon"]).is_file()


def test_embedded_metadata_is_quoted_and_matches_the_plugin_interface() -> None:
    metadata = _yaml_interface()
    raw_metadata = (SKILL_ROOT / "agents" / "openai.yaml").read_text(encoding="utf-8")
    manifest = json.loads((PLUGIN_ROOT / ".codex-plugin" / "plugin.json").read_text())

    assert set(metadata) == {"display_name", "short_description", "default_prompt"}
    assert all(f'{key}: "' in raw_metadata for key in metadata)
    assert metadata["display_name"] == manifest["interface"]["displayName"]
    assert metadata["short_description"] == manifest["interface"]["shortDescription"].rstrip(".")
    assert 25 <= len(metadata["short_description"]) <= 64
    assert "$echo" in metadata["default_prompt"]
    assert "locally with speaker labels" in metadata["default_prompt"]


def test_fork_attribution_is_preserved() -> None:
    notice = (PLUGIN_ROOT / "NOTICE.md").read_text(encoding="utf-8")
    license_text = (PLUGIN_ROOT / "LICENSE").read_text(encoding="utf-8")

    assert "olegvg/olegvg-skills" in notice
    assert "Oleg Gaidukov" in notice and "Oleg Gaidukov" in license_text
    assert "GigaAM" in notice
