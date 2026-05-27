
def store_output_details(outputs: list[dict]):
    return pd.DataFrame([{
                "prompt": prompt,
                "columns": columns,
                "k": k,
                **out,
            } for out in outputs])
    
def main():
    abstract_experiment()
    quit(0)
    
    score, outs = evaluate_qa(tests, qa, verbose=False)
    report.to_csv('report.csv')
    
    print('score:', score)


if __name__ == "__main__":
    main()