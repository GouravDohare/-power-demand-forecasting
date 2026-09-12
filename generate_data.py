from train import make_data

if __name__ == "__main__":
    df = make_data()
    df.to_csv("synthetic_demand.csv", index=False)
    print("Wrote synthetic_demand.csv")
