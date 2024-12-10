**Sharpe-Ratio-Maximizing-Portfolio-Generator**

--------------------------------------------------------------------------------------------------------------------------------
**Table of Contents:**

Info & How to Use

How it Works

Analysis Implications

Code Drawbacks

Future Projects

Sources

Updates

--------------------------------------------------------------------------------------------------------------------------------
**Info & How to Use:**

This code generates a stock portfolio that maximizes the Sharpe Ratio (((RoR of Portfolio) - (RoR of Risk-Free Asset))/Risk of Portfolio) based upon a specific time range. 

To see how the process works on a mathematical basis, please see (Source 1) and (Source 2): we are essentially doing that for all of the stocks in the modern-day S&P500.

**When editing, ONLY change the time of the _start_date_ and _end_date_.**

Files will appear after the code is done called "ouput" in the downloaded folder. 

**The ouput will consist of:**

Cleaned_Data.xls                           -> The cleaned daily stock prices of all stocks in the S&P500

Optimal_Portfolio_Results.xls              -> What stocks make up the Portfolio that maximizes the Sharpe Ratio

Portfolio_Holding_Analysis.xls             -> A comparision between the Relative % Change between the created portfolio and the S&P500 (work is shown, and graph is created as .png)

Portfolio_vs_SPY_Comparison.png            -> The graph that compares the S&P500 to the created portfolio(from end_date to today)

Rate_of_Return_and_Covariance.xls          -> The Daily Rate of Return and Covariance Table for all stocks in S&P500, used to create the Optimal Portfolio that'd maximize the Sharpe Ratio

--------------------------------------------------------------------------------------------------------------------------------
**Analysis Implications:**

The Portfolio_Holding_Analysis.xls file can allow the performance of the portfolio to compare with the S&P500. From this, a key factor can be found among all top-performing Sharpe Portfolios:

**Resilience is the greatest asset (for long-term returns)** - The portfolios formed right before a market boom (typically right after a market crash) typically have the highest outperformance of the S&P500 over the course of 2-5 years. Since these stocks that hadn't crashed significantly, they're show to other investors the company's resiliency, which will cause more and more people to invest in the company, as it is shown to be a more stable stock. Further research can be done on said analysis regarding how reliant said companies are on Valuation (i.e. earnings, EBITDA) vs Pricing (market outlook). 
This can be observed in a few examples:

- **Post 2008 Market Crash:** at the worst of the 2008 market crash, if a portfolio was formed at that time it would've underperformed for the first 2 years, but afterwards, it had consistently outperformed the S&P500, compounding over that time to offer a 3810% return in the course of 16 years, whereas there was only a 1100% return for the S&P500.
  
- **Post 2020 Market Crash:** at the worst of the 2020 market crash, if a portfolio was formed at that time, it would've ended up with a 200% by 2024, while the S&P500 only exhibited a 138% return.
  
- **Pre-2024 Market Boom:** Before the 2024 market boom, if a portfolio was formed during the 2024 market boom, it would've had a 41% return in the course of 6 months, whereas the S&P500 has only recieved 17% return.


--------------------------------------------------------------------------------------------------------------------------------
**Code Drawbacks**

Along with the implications mentioned, there are a few drawbacks to this code, being:

**1.) Yahoo Finance API** - the Yahoo Finace API malfunctions from time to time (due to maintenance activities), which results in occasional random errors/inconsistencies (i.e. inability to access stocks, Risk Free Rate, etc). To stop these errors from occuring, wait 30-60 minutes to rerun code, as maitenance from Yahoo Finance would have past.

**2.) The Limited Stock Data Source** - the current api being used is from Yahoo Finance (yfinance), which only contains the *present* S&P500 stocks, meaning that the code doesn't account for stocks that *left* the S&P500, due to bankruptcy, or stock decline.

**3.) The Risk Free Rate** - The Risk Free Rate is leveraged off of Yahoo Finance's API, which (along with Yahoo Finance's API maintenance), can have errors of its own (i.e. not being able to retrieve the present-day 13-week bond rate). 

--------------------------------------------------------------------------------------------------------------------------------
**Future Projects/Updates**

Updates to this Software are necessary to improve accuracy and maintain utility in the current financial climate today. Here is what will be done in the future:

**1.) Fix Data Sources** - Using @fja05680's S&P500 stock list over history can help to improve historical Sharpe Ratio usage & Backtesting (See Source 3).

**2.) Auto-Updating Data** - Creating a method to auto-update the data over a certain period of time can allow updates to the portfolio to be done passively, enabling the ability to trade

**3.) Improve Data Fallbacks** - Providing methods to prevent inconsistent/incorrect results during times of maintenance can allow for complete passive trading with the algorithim (i.e. the Risk Free Rate fallback rate of 5% can be improved upon).

**4.) Auto-Trading Capabilities** - Connecting this to an API with stock trading data that matches the Data Source can enable live trading to be done, and data to be made/saved.

**5.) Exit Strategy Capabilities** - As seen in many scenarios of analysis, sometimes there are peaks to the price difference between the Sharpe Portfolio and the S&P500. Timing the exit right could enable an even higher rate-of-return. 

**6.) Improve Speed** - Enabling quicker actions could allow this algorithim to day trade, and allow the algorithim a greater extent to be analyzed & improved.

**7.) Improve Analysis** - Through allowing the algorithim to run through stock prices over different time periods, we can determine optimal entry & exit strategies. 

**8.) Automate Learning Process** - Through an analysis workflow, we can allow the algorithim to learn & improve passively.

--------------------------------------------------------------------------------------------------------------------------------
**Sources**

1. https://www.youtube.com/watch?v=tMNPYRhMqos&t=851s&ab_channel=YongJinPark

2. https://www.youtube.com/watch?v=-V4lVvi5lL0&pp=ygUTeW9uZyBqaW4gcGFyayBleGNlbA%3D%3D

3. https://github.com/fja05680/sp500?

--------------------------------------------------------------------------------------------------------------------------------

**Updates**

12/9/2024 - nothing yet! Just wrapped up uploading the full project onto here! :)
