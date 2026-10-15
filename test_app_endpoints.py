import json
from web_app.app import app, initialize_app

def test_endpoints():
    initialize_app()
    client = app.test_client()

    print("\n1. Testing GET / ...")
    res = client.get("/")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    print("   Passed! (Status 200, HTML length:", len(res.data), ")")

    print("\n2. Testing GET /api/sample_cases ...")
    res = client.get("/api/sample_cases")
    assert res.status_code == 200
    samples = json.loads(res.data)
    print(f"   Passed! Found {len(samples)} sample cases.")
    assert len(samples) > 0

    print("\n3. Testing POST /api/classify ...")
    test_text = samples[0]['text_snippet']
    res = client.post("/api/classify", json={"text": test_text})
    assert res.status_code == 200
    pred = json.loads(res.data)
    print("   Passed! Classify output:")
    print("   Doc2Vec Prediction:", pred.get("doc2vec", {}).get("predicted_area_name"), "Confidence:", pred.get("doc2vec", {}).get("confidence"))
    print("   LDA Prediction:    ", pred.get("lda", {}).get("predicted_area_name"), "Confidence:", pred.get("lda", {}).get("confidence"))

    print("\n4. Testing GET /api/precedents ...")
    res = client.get("/api/precedents?doc_id=3397&topn=5")
    assert res.status_code == 200
    precedents = json.loads(res.data)
    print("   Passed! Precedents for Roe v. Wade:")
    for p in precedents.get("similar_precedents", []):
        print(f"     [{p['similarity_score']:.4f}] {p['case_name']} ({p['us_cite_id']}) - {p['issue_area_name']}")

    print("\n5. Testing GET /api/metrics ...")
    res = client.get("/api/metrics")
    assert res.status_code == 200
    metrics = json.loads(res.data)
    print("   Passed! Metrics dataset total:", metrics.get("dataset", {}).get("total_decisions"))

    print("\nALL ENDPOINT TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
