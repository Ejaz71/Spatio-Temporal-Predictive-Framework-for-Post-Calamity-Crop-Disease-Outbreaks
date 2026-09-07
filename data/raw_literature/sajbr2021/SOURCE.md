Source: Mahmud, Q. M., Rahman, A. H. M. A., Akanda, A. M., Hossain, M. M., and Hossain, M. M. (2021).
Survey on rice blast disease incidence in major rice growing areas of Bangladesh.
South Asian Journal of Biological Research (SAJBR), 4(1):1-13.
License: CC BY 4.0
URL: http://aiipub.com/journals/sajbr-210327-031164/

Coverage: 8 districts (Dinajpur, Rangpur, Bogura, Natore, Meherpur, Rajbari, Mymensingh, Jashore),
3 upazilas per district (24 upazilas total), 3 Boro seasons (2017-2019).
72 upazila x year records; aggregated to 24 district x year records in ../events.csv.

Outbreak threshold: leaf blast incidence >= 15% OR neck blast incidence >= 8%
(district-level aggregate values used for the threshold, per proposal Section 6.3).
8 positive / 16 negative district-year events at this threshold.

`upazila_x_year_raw.csv` in this folder is the finer-grained upazila-level source table
(72 rows) behind the district-year aggregates in `../events.csv`. Not currently used as the
primary modeling unit (proposal Section 6.2 specifies district-season events as Option A),
but retained as a real, citable higher-resolution dataset for a future robustness check.
