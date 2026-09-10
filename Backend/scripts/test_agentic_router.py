from agent.router import route_query


def test(
    name,
    query,
    images,
    modalities=None,
):

    result = route_query(
        query=query,
        number_of_images=images,
        modalities=modalities,
    )

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    print("Query:")
    print(query)

    print("\nImages:")
    print(images)

    print("\nModalities:")
    print(modalities)

    print("\nSelected Task:")
    print(result["task"])

    print("\nRouting Confidence:")
    print(result["confidence"])

    print("\nReason:")
    print(result["reason"])

    print("\nCandidate Scores:")

    for task, score in result["candidates"].items():
        print(f"  {task}: {score}")


def main():

    test(
        name="SINGLE IMAGE VQA",
        query="What type of land cover is visible?",
        images=1,
    )

    test(
        name="IMAGE CAPTIONING",
        query="Describe the satellite scene.",
        images=1,
    )

    test(
        name="CHANGE VQA",
        query="What changed between these two images?",
        images=2,
    )

    test(
        name="CHANGE DETECTION",
        query="Detect changes between these images.",
        images=2,
    )

    test(
        name="OPTICAL + SAR",
        query="Analyze these optical and SAR images together.",
        images=2,
        modalities=[
            "optical",
            "sar",
        ],
    )


if __name__ == "__main__":
    main()