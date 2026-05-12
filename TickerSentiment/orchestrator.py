import yfinance as yf
from agents.data_agent    import DataAgent
from agents.finbert_agent import FinBERTAgent
from agents.llm_agent     import OllamaAnalystAgent, OllamaValidatorAgent
 
def resolve_company(ticker: str) -> str:
    try:
        return yf.Ticker(ticker).info.get('longName', ticker)
    except Exception:
        return ticker
 
def analyze_ticker(ticker: str) -> dict:
    company = resolve_company(ticker)
    print(f'[{ticker}] -> {company}')
 
    social  = DataAgent().fetch(ticker, company)
    finbert = FinBERTAgent().score(social.combined_text)
    llm     = OllamaAnalystAgent().analyze(ticker, company, social.combined_text)
    val     = OllamaValidatorAgent().validate(ticker, llm, social.combined_text)
 
    final_score = (
        llm['sentiment_score']    * 0.50
        + val['validator_score']  * 0.30
        + finbert['finbert_score']* 0.20
    )
    return {
        'ticker':      ticker,
        'company':     company,
        'final_score': round(final_score, 3),
        'final_label': llm['sentiment_label'],
        'consensus':   val['agree'],
        'llm':         llm,
        'validation':  val,
        'finbert':     finbert,
    }
 
def main(tickers: list[str]):
    results = [analyze_ticker(t) for t in tickers]
    print('\n' + '='*55)
    for r in results:
        consensus = 'Yes' if r['consensus'] else 'No'
        print(f"{r['ticker']:<8} {r['final_score']:>+.3f}  "
              f"{r['final_label']:<10}  consensus={consensus}")
 
if __name__ == '__main__':
    main(['AAPL', 'NVDA', 'TSLA'])
