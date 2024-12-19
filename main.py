#TYPE "PIP INSTALL ______" all packages here, on the terminal on VS Studio:
#The terminal is on the bottom part of the VS Studio Screen
#https://www.youtube.com/watch?v=ThU13tikHQw&ab_channel=AdityaThakur
#Use that URL if you're still Struggling

import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from datetime import datetime, timedelta
from pytickersymbols import PyTickerSymbols
import os
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


#______________Portfolio Optimization Object (Ignore & Scroll to Bottom!)________________#



class PortfolioOptimizer:
    def __init__(self, start_date: str, end_date: str):
            """
            Initialize the Portfolio Optimizer with a date range.
            :param start_date: Start date for historical stock data.
            :param end_date: End date for historical stock data.
            """
            self.start_date = start_date
            self.end_date = end_date
            self.rfr = None
            self.returns_data = None
            self.cov_matrix = None
            self.optimal_weights = None
            self.stock_symbols = None

            # Set base directory and output folder
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
            self.output_folder = os.path.join(self.base_dir, "output")
            
            # Default paths for output files
            self.cleaned_file_path = os.path.join(self.output_folder, "Cleaned_Data.csv")
            self.excel_output_path = os.path.join(self.output_folder, "Rate_of_Return_and_Covariance.xlsx")
            self.results_csv_path = os.path.join(self.output_folder, "Optimal_Portfolio_Results.csv")
            
            # Ensure the output directory exists
            os.makedirs(self.output_folder, exist_ok=True)

            # Set the date of optimization
            self.date_of_optimization = datetime.now().strftime("%Y-%m-%d")

    def fetch_data(self, tickers: list):
        print("Fetching data...")
        data = yf.download(tickers, start=self.start_date, end=self.end_date, interval="1d", progress=False)
        
        # Extract 'Adj Close' prices explicitly
        adj_close_data = data.get("Adj Close", pd.DataFrame())
        print(f"Columns before dropna: {adj_close_data.columns}")
        
        if adj_close_data.empty:
            raise ValueError("No valid data fetched for the provided tickers.")
        
        # Drop columns with all NaN values
        adj_close_data.dropna(axis=1, how="all", inplace=True)
        
        # Update stock symbols
        self.stock_symbols = adj_close_data.columns.tolist()
        print(f"Stock symbols fetched: {self.stock_symbols}")
        
        if not self.stock_symbols:
            raise ValueError("No valid stock symbols found. Check your data source or ticker list.")
        
        # Clean and save the data
        self.clean_data(adj_close_data)



    def clean_data(self, data: pd.DataFrame):
        """
        Clean the fetched data by ensuring a proper timestamp column and removing irrelevant columns.
        :param data: Raw DataFrame fetched from Yahoo Finance.
        """
        # Ensure the index is reset only if needed
        if isinstance(data.index, pd.DatetimeIndex):
            if "timestamp" not in data.columns:  # Avoid creating duplicate timestamp columns
                data = data.reset_index()  # Reset index to make the datetime index a column
                data.rename(columns={"index": "timestamp"}, inplace=True)
            else:
                print("'timestamp' column already exists. Skipping index reset.")
        elif "timestamp" not in data.columns:
            raise KeyError("The 'timestamp' column is missing. Ensure the index is properly reset or renamed.")
    
        # After resetting the index, ensure the 'timestamp' column is named correctly
        if "Date" in data.columns:
            data.rename(columns={"Date": "timestamp"}, inplace=True)
    
        # Validate the presence of the 'timestamp' column
        if "timestamp" not in data.columns:
            raise KeyError("The 'timestamp' column is missing after processing. Check the raw data structure.")
    
        # Remove duplicate columns explicitly
        data = data.loc[:, ~data.columns.duplicated()]
    
        # Identify columns to keep where the first row has non-NaN values
        columns_to_keep = ['timestamp'] + [col for col in data.columns if data[col].notna().iloc[0]]
    
        # Filter the DataFrame to retain only the desired columns
        cleaned_data = data[columns_to_keep]
    
        # Save the cleaned data to a CSV file
        os.makedirs(os.path.dirname(self.cleaned_file_path), exist_ok=True)
        cleaned_data.to_csv(self.cleaned_file_path, index=False)
        print(f"Cleaned data saved to {self.cleaned_file_path}")
    
        # Assign cleaned data to the class attribute
        self.cleaned_data = cleaned_data
    
    def calculate_risk_free_rate(self):
        """
        Fetch the most recent valid 13-week U.S. Treasury Bill yield (^IRX).
        If not available, loop backward in time until data is found.
        """
        print("Fetching risk-free rate...")
        t_bill = yf.Ticker("^IRX")
        daily_rfr = None
        days_to_check = 1  # Start with one day back
        max_attempts = 30  # Limit search to the last 30 days
    
        for attempt in range(max_attempts):
            try:
                t_bill_history = t_bill.history(period=f"{days_to_check}d")
                if not t_bill_history.empty:
                    # Get the most recent yield
                    current_yield_annual = t_bill_history["Close"].dropna().iloc[-1]
                    daily_rfr = (current_yield_annual / 252) / 100  # Convert annual yield to daily rate
                    print(f"Risk-free rate found: {daily_rfr:.6f}")
                    break
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
            days_to_check += 1
    
        if daily_rfr is None:
            fallback_rate = 5.0  # Default annual fallback rate (5%)
            daily_rfr = (fallback_rate / 252) / 100
            print(f"Warning: Using fallback risk-free rate of {fallback_rate}% (annual).")
    
        self.rfr = daily_rfr
        print(f"Final Daily Risk-Free Rate: {self.rfr:.6f}")
    
    
        
    def calculate_rate_of_return(self):
        """
        Calculate daily rate of return from the cleaned data.
        """
        # Load cleaned data
        data = pd.read_csv(self.cleaned_file_path)
        
        # Remove duplicate columns
        if "timestamp" in data.columns:
            data.drop(columns=["timestamp"], inplace=True)
            print("Duplicate 'timestamp' column removed.")
        
        # Convert numeric columns
        numeric_data = data.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
        
        # Calculate daily rate of return
        daily_ror = numeric_data.pct_change().dropna()
        self.returns_data = daily_ror.mean()  # Store average daily returns
        
        # Update stock symbols to match valid returns
        self.stock_symbols = daily_ror.columns.tolist()
        print(f"Updated stock symbols to match returns data: {len(self.stock_symbols)} symbols")
    


    def calculate_covariance_matrix(self):
        """
        Calculate the covariance matrix from the cleaned data, removing any duplicate or unnecessary columns.
        """
        # Load cleaned data
        data = pd.read_csv(self.cleaned_file_path)
        
        # Remove duplicate columns explicitly
        if "timestamp" in data.columns:
            data.drop(columns=["timestamp"], inplace=True)
            print("Duplicate 'timestamp' column removed.")
    
        # Ensure only numeric columns are used for calculations
        numeric_data = data.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    
        # Calculate covariance matrix
        self.cov_matrix = numeric_data.pct_change().dropna().cov()
        print("Covariance matrix calculated successfully.")
    

    def calculate_portfolio_metrics(self, weights):
        """
        Calculate portfolio metrics for given weights.
        :param weights: Array of weights for each stock.
        :return: Dictionary with portfolio metrics.
        """
        portfolio_return = np.dot(self.returns_data, weights)
        portfolio_variance = np.dot(weights.T, np.dot(self.cov_matrix, weights))
        portfolio_stdev = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.rfr) / portfolio_stdev

        return {
            "Return": portfolio_return,
            "Risk": portfolio_variance,
            "StDev": portfolio_stdev,
            "Sharpe Ratio": sharpe_ratio,
        }

    def optimize_weights(self):
        if self.returns_data is None or self.cov_matrix is None:
            raise ValueError("Error: Returns data or covariance matrix is not calculated. Please ensure proper execution order.")
        
        num_assets = len(self.returns_data)
        initial_weights = np.ones(num_assets) / num_assets  # Equal weights initially
        bounds = [(0, 1) for _ in range(num_assets)]  # Weights between 0 and 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})  # Weights sum to 1
    
        # Objective function to minimize (negative Sharpe Ratio)
        def negative_sharpe(weights):
            metrics = self.calculate_portfolio_metrics(weights)
            return -metrics["Sharpe Ratio"]
    
        result = minimize(negative_sharpe, initial_weights, bounds=bounds, constraints=constraints)
    
        if result.success:
            self.optimal_weights = result.x
            print(f"Number of optimized weights: {len(self.optimal_weights)}")
            print("Optimization successful.")
        else:
            raise ValueError("Optimization failed.")
    


    def get_results(self):
        if self.optimal_weights is None:
            raise ValueError("Optimal weights are not calculated. Ensure optimization is completed successfully.")
        
        if len(self.stock_symbols) != len(self.optimal_weights):
            raise ValueError(
                f"Mismatch in lengths: {len(self.stock_symbols)} stock symbols vs. {len(self.optimal_weights)} weights."
            )
        
        weights_df = pd.DataFrame({
            "Symbol": self.stock_symbols,
            "Weight (%)": self.optimal_weights * 100
        })
        weights_df = weights_df[weights_df["Weight (%)"] > 1e-5]  # Filter significant weights
        weights_df = weights_df.sort_values(by="Weight (%)", ascending=False)
        weights_df["Date"] = self.date_of_optimization
        return weights_df
    def export_to_excel(self, output_path=None):
        """
        Export Rate of Return and Covariance Matrix to an Excel file.
        :param output_path: Optional path to save the Excel file. If not provided, uses the class attribute.
        """
        # Use the provided path or the class attribute path
        path = output_path or self.excel_output_path
        with pd.ExcelWriter(path) as writer:
            # Export rate of return
            self.returns_data.to_frame(name="Average Daily Returns").to_excel(writer, sheet_name="Rate of Return")

            # Export covariance matrix
            self.cov_matrix.to_excel(writer, sheet_name="Covariance Matrix")
        print(f"Excel file saved at {path}")

    def set_output_paths(self, cleaned_file_path=None, excel_output_path=None, results_csv_path=None):
            """
            Allow users to set custom output paths for the cleaned data, Excel file, and results CSV.
            :param cleaned_file_path: Path to save the cleaned data.
            :param excel_output_path: Path to save the Excel file with RoR and covariance matrix.
            :param results_csv_path: Path to save the final portfolio results.
            """
            if cleaned_file_path:
                self.cleaned_file_path = os.path.join(self.output_folder, cleaned_file_path)
                print(f"Cleaned data path set to: {self.cleaned_file_path}")

            if excel_output_path:
                self.excel_output_path = os.path.join(self.output_folder, excel_output_path)
                print(f"Excel output path set to: {self.excel_output_path}")

            if results_csv_path:
                self.results_csv_path = os.path.join(self.output_folder, results_csv_path)
                print(f"Results CSV path set to: {self.results_csv_path}")

    def analyze_portfolio_vs_spy(self):
        """
        Fetch adjusted close prices for portfolio stocks and SPY based on Optimal Portfolio Results,
        calculate % change, weighted % change, and portfolio vs SPY % change comparison.
        Save all results in a new Excel file, including Portfolio and SPY metrics.
        """
        # Load the optimal portfolio results to get the relevant stocks
        optimal_results = pd.read_csv(self.results_csv_path)
        relevant_stocks = optimal_results["Symbol"].tolist()  # List of stocks with significant weights

        # Include SPY for comparison
        relevant_stocks.append("SPY")

        # Fetch adjusted close prices for relevant stocks
        print(f"Fetching adjusted close prices for {relevant_stocks}...")
        prices = yf.download(relevant_stocks, start=self.end_date, end=datetime.today().strftime('%Y-%m-%d'), interval="1d", progress=False)["Adj Close"]

        # Check if data is fetched
        if prices.empty:
            raise ValueError("No data fetched. Please check the ticker symbols or date range.")

        # Convert index to timezone-unaware datetime
        prices.index = prices.index.tz_localize(None)

        # Calculate daily % change as numbers (not percentages)
        pct_change = (prices - prices.iloc[0]) / prices.iloc[0]  # % change relative to the first day's price
        pct_change.iloc[0] = 0  # Ensure the first day starts at 0

        # Extract portfolio weights for relevant stocks (excluding SPY)
        weights_df = optimal_results.set_index("Symbol")
        portfolio_weights = weights_df.loc[relevant_stocks[:-1], "Weight (%)"] / 100  # Convert % to proportions

        # Calculate weighted daily % change for the portfolio
        weighted_pct_change = pct_change[relevant_stocks[:-1]].mul(portfolio_weights, axis=1)  # Multiply % change by weights
        portfolio_daily_pct_change = weighted_pct_change.sum(axis=1)  # Sum weighted % change for portfolio

        # Create a DataFrame comparing Portfolio % Change and SPY % Change
        pct_change_comparison = pd.DataFrame({
            "Portfolio % Change": portfolio_daily_pct_change * 100,  # Convert to percentage values
            "SPY % Change": pct_change["SPY"] * 100  # Convert to percentage values
        })

        # Calculate Portfolio Metrics
        portfolio_metrics = self.calculate_portfolio_metrics(np.array(self.optimal_weights))

        # Calculate SPY Metrics (Single stock)
        spy_weights = np.zeros(len(self.stock_symbols))
        spy_index = self.stock_symbols.index("SPY")
        spy_weights[spy_index] = 1.0
        spy_metrics = self.calculate_portfolio_metrics(spy_weights)

        # Add Portfolio and SPY metrics to a new DataFrame
        metrics_comparison = pd.DataFrame({
            "Metric": ["Return", "Risk", "StDev", "Sharpe Ratio"],
            "Portfolio": [
                portfolio_metrics["Return"],
                portfolio_metrics["Risk"],
                portfolio_metrics["StDev"],
                portfolio_metrics["Sharpe Ratio"],
            ],
            "SPY": [
                spy_metrics["Return"],
                spy_metrics["Risk"],
                spy_metrics["StDev"],
                spy_metrics["Sharpe Ratio"],
            ],
        })

        # Output file path for the new Excel file
        analysis_output_path = os.path.join(self.output_folder, "Portfolio_Holding_Analysis.xlsx")

        # Write everything to a new Excel file
        with pd.ExcelWriter(analysis_output_path, engine='openpyxl') as writer:
            prices.to_excel(writer, sheet_name="Daily Adjusted Prices")
            pct_change.to_excel(writer, sheet_name="Relative % Change (Numbers)")
            weighted_pct_change.to_excel(writer, sheet_name="Weighted % Change (Portfolio)")
            pct_change_comparison.to_excel(writer, sheet_name="Relative % Change Comparison")
            metrics_comparison.to_excel(writer, sheet_name="Portfolio and SPY Metrics", index=False)

        print(f"Portfolio analysis saved to {analysis_output_path}")

        # Create and save a graph comparing Portfolio % Change and SPY % Change
        plt.figure(figsize=(10, 6))
        plt.plot(pct_change_comparison.index, pct_change_comparison["Portfolio % Change"], label="Portfolio % Change")
        plt.plot(pct_change_comparison.index, pct_change_comparison["SPY % Change"], label="SPY % Change")
        plt.xlabel("Date")
        plt.ylabel("Percentage Change (%)")
        plt.title("Portfolio vs SPY % Change Over Time")
        plt.legend()
        graph_output_path = os.path.join(self.output_folder, "Portfolio_vs_SPY_Comparison.png")
        plt.savefig(graph_output_path)
        plt.close()
        plt.show()
        print(f"Graph saved to {graph_output_path}")











#_____________________CODE_TO_USE_______________________________________#



# Get the absolute path of the script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the "output" folder
output_folder = os.path.join(base_dir, "output")


#date = "Year-month-date"
start_date = "2014-12-15"
end_date = "2018-12-15"


# Initialize optimizer
optimizer = PortfolioOptimizer(start_date=start_date, end_date=end_date)


# Set paths relative to the "output" folder
cleaned_file_path = os.path.join(output_folder, "Cleaned_Data.csv")
excel_output_path = os.path.join(output_folder, "Rate_of_Return_and_Covariance.xlsx")
results_csv_path = os.path.join(output_folder, "Optimal_Portfolio_Results.csv")

# Set the output paths in the optimizer
optimizer.set_output_paths(
    cleaned_file_path=cleaned_file_path,
    excel_output_path=excel_output_path,
    results_csv_path=results_csv_path
)


# Initialize PyTickerSymbols to fetch S&P 500 tickers
# Include SPY in the list of tickers
stock_data = PyTickerSymbols()
sp500_stocks = stock_data.get_stocks_by_index('S&P 500')
tickers = [stock['symbol'].replace('-', '.') for stock in sp500_stocks]  # Replace '-' with '.' for compatibility
tickers.append('SPY')


# Fetch data for the tickers
optimizer.fetch_data(tickers)

#Find Risk Free Rate @ End_Date
optimizer.calculate_risk_free_rate()

# Calculate rate of return
optimizer.calculate_rate_of_return()

# Calculate covariance matrix
optimizer.calculate_covariance_matrix()

# Export Rate of Return and Covariance Matrix
optimizer.export_to_excel()

# Optimize portfolio weights
optimizer.optimize_weights()

# Calculate and display portfolio metrics
metrics = optimizer.calculate_portfolio_metrics(optimizer.optimal_weights)
print("Portfolio Metrics:")
print(metrics)




print(f"Number of stock symbols: {len(optimizer.stock_symbols)}")
print(f"Number of optimized weights: {len(optimizer.optimal_weights)}")


# Get results and save to CSV
portfolio_results = optimizer.get_results()
# Save portfolio results to the specified CSV path
portfolio_results.to_csv(optimizer.results_csv_path, index=False)
print(f"Portfolio results saved to {optimizer.results_csv_path}")

# Analyze portfolio performance against SPY
optimizer.analyze_portfolio_vs_spy()

