Using Python for Blog Analysis
==============================

Source code for weblog collection & analysis using statcounter

- Use browser automation tools like iMacros(for firefox) to automate downloading the log file and schedule the script to run every week(required frequency). This helps to trick around the 500 row limit for free account.

- combine.py is a python script which aggregates data from multiple weblog files created through statcounter. It also creates a "user" field which identifies based on OS & IP Addr. Finally exports the combined file as a csv.

- Use excel/other spreadsheet tool to pivot the data for subsequent analysis.

Be data driven! Happy Blogging !!
