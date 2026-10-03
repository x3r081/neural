import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from neural_runtime.advisor_ui import _make_server


def test_advisor_http_page_and_injected_report_factory():
    calls = []
    report = {
        "generated_at_utc": "2026-10-03T00:00:00Z",
        "recommendations": [{"id": "candidate", "name": "Local candidate"}],
        "excluded": [],
        "notices": [],
    }

    def factory(**kwargs):
        calls.append(kwargs)
        return report

    server = _make_server(factory, "127.0.0.1", 0, {
        "context": 262144,
        "workload": "general",
        "scenario": "current",
        "cpu_bandwidth_gbps": 36.5,
    })
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urlopen(base + "/page") as response:
            page = response.read().decode("utf-8")
            assert response.status == 200
            assert "Neural model advisor" in page
            assert "Scan recommendations" in page
            assert "Advanced bandwidth assumptions" in page
            assert "CPU-expert bandwidth scenario" in page
            assert "fetch(`/api/recommend?${q}`)" in page
            assert 'max="100000"' in page
            assert 'value="262144"' not in page  # custom options are added through the safe DOM path
            assert '"context": 262144' in page

        with urlopen(base + "/api/recommend?context=32768&workload=general&scenario=current&cpu_bandwidth_gbps=36.5") as response:
            received = json.loads(response.read())
            assert response.status == 200
            assert received == report
        assert calls == [{
            "context": 32768,
            "workload": "general",
            "scenario": "current",
            "cpu_bandwidth_gbps": 36.5,
            "gpu_bandwidth_gbps": None,
        }]

        for query, message in [
            ("context=bad", "context must be a whole number of tokens"),
            ("context=1048577", "context must be an integer from 128 to 1048576 tokens"),
            ("cpu_bandwidth_gbps=0.09", "cpu_bandwidth_gbps must be between 0.1 and 100000 GB/s"),
            ("gpu_bandwidth_gbps=100001", "gpu_bandwidth_gbps must be between 0.1 and 100000 GB/s"),
        ]:
            try:
                urlopen(base + "/api/recommend?" + query)
                assert False, f"invalid query {query} should return HTTP 400"
            except HTTPError as error:
                assert error.code == 400
                assert json.loads(error.read())["error"] == message
        assert len(calls) == 1

        for headers in [
            {"Host": "example.com"},
            {"Origin": "http://example.com"},
            {"Origin": "http://127.0.0.1:1"},
        ]:
            try:
                urlopen(Request(base + "/api/recommend", headers=headers))
                assert False, "non-local request headers should return HTTP 403"
            except HTTPError as error:
                assert error.code == 403
        assert len(calls) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
