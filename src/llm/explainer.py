"""
LLM-based portfolio explainer.

Uses language models to explain portfolio allocation decisions
in natural language.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PortfolioExplainer:
    """
    Generate natural language explanations of portfolio allocations.
    
    Can use OpenAI API or work with mock explanations for demonstration.
    
    Parameters
    ----------
    api_key : str, optional
        OpenAI API key
    model : str
        Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
    use_mock : bool
        Use mock explanations (for testing without API)
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = 'gpt-4',
        use_mock: bool = True
    ):
        self.api_key = api_key
        self.model = model
        self.use_mock = use_mock
        
        if not use_mock and api_key:
            try:
                import openai
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model: {model}")
            except ImportError:
                logger.warning("openai package not installed, using mock mode")
                self.use_mock = True
        else:
            self.use_mock = True
            logger.info("Using mock explanation mode")
    
    def explain_allocation(
        self,
        weights: pd.Series,
        returns_stats: pd.Series,
        risk_stats: pd.Series,
        method: str = "QAOA"
    ) -> str:
        """
        Explain portfolio allocation.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights (ticker -> weight)
        returns_stats : pd.Series
            Expected returns per asset
        risk_stats : pd.Series
            Volatilities per asset
        method : str
            Optimization method used
            
        Returns
        -------
        str
            Natural language explanation
        """
        # Filter to selected assets
        selected = weights[weights > 0.001].sort_values(ascending=False)
        
        if self.use_mock:
            return self._mock_explanation(selected, returns_stats, risk_stats, method)
        else:
            return self._llm_explanation(selected, returns_stats, risk_stats, method)
    
    def _mock_explanation(
        self,
        weights: pd.Series,
        returns: pd.Series,
        risks: pd.Series,
        method: str
    ) -> str:
        """Generate rule-based explanation."""
        
        explanation = f"Portfolio Allocation Explanation ({method} Optimization)\n\n"
        
        # Overall strategy
        n_assets = len(weights)
        max_weight_ticker = weights.idxmax()
        max_weight_value = weights.max()
        
        explanation += f"This portfolio selects {n_assets} assets from the available universe. "
        explanation += f"The allocation is {max_weight_value:.1%} concentrated in {max_weight_ticker}, "
        explanation += "suggesting a growth-oriented strategy.\n\n"
        
        # Individual positions
        explanation += "Position Analysis:\n"
        for ticker, weight in weights.items():
            ret = returns.loc[ticker]
            risk = risks.loc[ticker]
            
            explanation += f"\n• {ticker} ({weight:.1%} allocation):\n"
            explanation += f"  - Expected return: {ret:.1%} annually\n"
            explanation += f"  - Volatility: {risk:.1%}\n"
            
            if ret > returns.median():
                explanation += "  - Above-average return potential\n"
            if risk < risks.median():
                explanation += "  - Below-average risk profile\n"
            
            if weight == weights.max():
                explanation += "  - Largest position: Conviction pick for growth\n"
            elif weight < 0.15:
                explanation += "  - Smaller position: Diversification play\n"
        
        # Risk-return tradeoff
        avg_return = (weights * returns).sum()
        explanation += f"\n\nPortfolio Characteristics:\n"
        explanation += f"Expected Return: {avg_return:.1%} annually\n"
        explanation += f"Strategy: Balance between high-growth potential and risk management\n"
        
        if method == "QAOA":
            explanation += "\nQuantum Advantage:\n"
            explanation += "This allocation was discovered using Quantum Approximate Optimization "
            explanation += "Algorithm (QAOA), which efficiently explored the discrete solution space "
            explanation += "of asset combinations to find a near-optimal portfolio configuration.\n"
        
        return explanation
    
    def _llm_explanation(
        self,
        weights: pd.Series,
        returns: pd.Series,
        risks: pd.Series,
        method: str
    ) -> str:
        """Generate LLM-based explanation using OpenAI API."""
        
        # Prepare data summary
        data_summary = "Portfolio allocation:\n"
        for ticker, weight in weights.items():
            data_summary += f"- {ticker}: {weight:.1%} (Return: {returns.loc[ticker]:.1%}, Risk: {risks.loc[ticker]:.1%})\n"
        
        prompt = f"""You are a financial advisor explaining a portfolio allocation to a client.

{data_summary}

This portfolio was optimized using {method}, a {
'quantum computing algorithm' if method == 'QAOA' else 'classical optimization method'
}.

Provide a clear, professional explanation of:
1. Overall strategy and allocation approach
2. Key positions and their rationale
3. Risk-return characteristics
4. Why this allocation makes sense

Keep the explanation concise (200-300 words) and accessible to investors.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert financial advisor."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return self._mock_explanation(weights, returns, risks, method)
    
    def compare_portfolios(
        self,
        classical_weights: pd.Series,
        quantum_weights: pd.Series,
        returns: pd.Series,
        classical_sharpe: float,
        quantum_sharpe: float
    ) -> str:
        """
        Compare classical vs quantum portfolio solutions.
        
        Parameters
        ----------
        classical_weights : pd.Series
            Classical portfolio weights
        quantum_weights : pd.Series
            Quantum portfolio weights
        returns : pd.Series
            Expected returns
        classical_sharpe : float
            Classical Sharpe ratio
        quantum_sharpe : float
            Quantum Sharpe ratio
            
        Returns
        -------
        str
            Comparison explanation
        """
        if self.use_mock:
            comparison = "Classical vs Quantum Portfolio Comparison\n\n"
            
            # Asset selection differences
            classical_assets = set(classical_weights[classical_weights > 0.001].index)
            quantum_assets = set(quantum_weights[quantum_weights > 0.001].index)
            
            common = classical_assets & quantum_assets
            only_classical = classical_assets - quantum_assets
            only_quantum = quantum_assets - classical_assets
            
            comparison += f"Asset Selection:\n"
            comparison += f"• Common holdings: {', '.join(common) if common else 'None'}\n"
            comparison += f"• Classical-only: {', '.join(only_classical) if only_classical else 'None'}\n"
            comparison += f"• Quantum-only: {', '.join(only_quantum) if only_quantum else 'None'}\n\n"
            
            # Performance
            comparison += f"Performance Metrics:\n"
            comparison += f"• Classical Sharpe Ratio: {classical_sharpe:.3f}\n"
            comparison += f"• Quantum Sharpe Ratio: {quantum_sharpe:.3f}\n"
            
            if quantum_sharpe > classical_sharpe:
                diff = (quantum_sharpe - classical_sharpe) / classical_sharpe * 100
                comparison += f"• Quantum advantage: +{diff:.1f}% better Sharpe ratio\n\n"
                comparison += "The quantum approach discovered a superior solution "
                comparison += "by efficiently exploring the combinatorial space of portfolio configurations.\n"
            else:
                comparison += "\nBoth approaches found high-quality solutions. "
                comparison += "The difference illustrates the trade-offs between exact optimization "
                comparison += "and quantum approximation methods.\n"
            
            return comparison
        else:
            # Could implement LLM-based comparison here
            return self.compare_portfolios(
                classical_weights, quantum_weights, returns,
                classical_sharpe, quantum_sharpe
            )


def generate_investment_thesis(
    ticker: str,
    weight: float,
    expected_return: float,
    volatility: float
) -> str:
    """
    Generate simple investment thesis for an asset.
    
    Parameters
    ----------
    ticker : str
        Asset ticker
    weight : float
        Portfolio weight
    expected_return : float
        Expected return
    volatility : float
        Volatility
        
    Returns
    -------
    str
        Investment thesis
    """
    thesis = f"Investment Thesis: {ticker}\n"
    thesis += f"Position Size: {weight:.1%}\n\n"
    
    if weight > 0.25:
        thesis += "CONVICTION BUY: Large position reflects high confidence "
        thesis += "in this asset's return potential.\n"
    elif weight < 0.10:
        thesis += "DIVERSIFICATION PLAY: Smaller position for portfolio balance.\n"
    else:
        thesis += "CORE HOLDING: Moderate position balancing return and risk.\n"
    
    thesis += f"\nExpected annual return: {expected_return:.1%}\n"
    thesis += f"Volatility: {volatility:.1%}\n"
    
    if expected_return / volatility > 0.5:
        thesis += "\nFavorable risk-reward profile supports this allocation.\n"
    
    return thesis


if __name__ == "__main__":
    # Test explainer
    weights = pd.Series({
        'AAPL': 0.30,
        'MSFT': 0.25,
        'NVDA': 0.25,
        'GOOGL': 0.15,
        'META': 0.05
    })
    
    returns = pd.Series({
        'AAPL': 0.12,
        'MSFT': 0.15,
        'NVDA': 0.25,
        'GOOGL': 0.14,
        'META': 0.10
    })
    
    risks = pd.Series({
        'AAPL': 0.22,
        'MSFT': 0.20,
        'NVDA': 0.35,
        'GOOGL': 0.24,
        'META': 0.28
    })
    
    explainer = PortfolioExplainer(use_mock=True)
    explanation = explainer.explain_allocation(weights, returns, risks, method="QAOA")
    
    print(explanation)
