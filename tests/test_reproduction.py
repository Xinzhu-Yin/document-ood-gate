from document_ood_gate.reproduction import classify_reproduction_delta


def test_reproduction_traffic_lights() -> None:
    assert classify_reproduction_delta(0.01, 0.02) == "green"
    assert classify_reproduction_delta(0.02, 0.03) == "yellow"
    assert classify_reproduction_delta(0.031, 0.00) == "red"
    assert classify_reproduction_delta(0.00, 0.051) == "red"

