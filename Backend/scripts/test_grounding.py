from models.grounding import get_grounding_model


IMAGE_PATH = "data/samples/test.jpg"


def main():

    print("=" * 60)
    print("SATQUERY AI - GROUNDING TEST")
    print("=" * 60)

    model = get_grounding_model()

    result = model.predict(
        image=IMAGE_PATH,
        query="building"
    )

    print("\nGrounding Result")
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()