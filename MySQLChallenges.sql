use tennis;

create table us_men_matches (player varchar(250), matches int);
insert into us_men_matches select player1,count(player1) as matches from us_men_2013 group by player1 union select player2,count(player2) as matches from us_men_2013 group by player2;
select player, sum(matches) as totalmatches from us_men_matches group by player order by totalmatches desc limit 5;
+--------------------+--------------+
| player             | totalmatches |
+--------------------+--------------+
| Rafael Nadal       |            7 |
| Richard Gasquet    |            6 |
| Stanislas Wawrinka |            6 |
| Novak Djokovic     |            6 |
| Tommy Robredo      |            5 |
+--------------------+--------------+

create table us_women_matches (player varchar(250), matches int);
insert into us_women_matches select player1,count(player1) as matches from us_women_2013 group by player1 union select player2,count(player2) as matches from us_women_2013 group by player2;
select player, sum(matches) as totalmatches from us_women_matches group by player order by totalmatches desc limit 5;
+------------+--------------+
| player     | totalmatches |
+------------+--------------+
| V Azarenka |            7 |
| N Li       |            6 |
| S Williams |            6 |
| F Pennetta |            5 |
| A Ivanovic |            4 |
+------------+--------------+

create table aus_men_matches (player varchar(250), matches int);
insert into aus_men_matches select player1,count(player1) as matches from aus_men_2013 group by player1 union select player2,count(player2) as matches from aus_men_2013 group by player2;
select player, sum(matches) as totalmatches from aus_men_matches group by player order by totalmatches desc limit 5;
+--------------------+--------------+
| player             | totalmatches |
+--------------------+--------------+
| Rafael Nadal       |            7 |
| Stanislas Wawrinka |            6 |
| Tomas Berdych      |            6 |
| Roger Federer      |            6 |
| Andy Murray        |            5 |
+--------------------+--------------+

create table aus_women_matches (player varchar(250), matches int);
insert into aus_women_matches select player1,count(player1) as matches from aus_women_2013 group by player1 union select player2,count(player2) as matches from aus_women_2013 group by player2;
select player, sum(matches) as totalmatches from aus_women_matches group by player order by totalmatches desc limit 5;
+---------------------+--------------+
| player              | totalmatches |
+---------------------+--------------+
| Dominika Cibulkova  |            7 |
| Na Li               |            7 |
| Agnieszka Radwanska |            6 |
| Eugenie Bouchard    |            6 |
| Ana Ivanovic        |            5 |
+---------------------+--------------+

create table french_men_matches (player varchar(250), matches int);
insert into french_men_matches select player1,count(player1) as matches from french_men_2013 group by player1 union select player2,count(player2) as matches from french_men_2013 group by player2;
select player, sum(matches) as totalmatches from french_men_matches group by player order by totalmatches desc limit 5;
+--------------------+--------------+
| player             | totalmatches |
+--------------------+--------------+
| David Ferrer       |            7 |
| Rafael Nadal       |            7 |
| Novak Djokovic     |            6 |
| Jo-Wilfried Tsonga |            6 |
| Tommy Haas         |            5 |
+--------------------+--------------+

create table french_women_matches (player varchar(250), matches int);
insert into french_women_matches select player1,count(player1) as matches from french_women_2013 group by player1 union select player2,count(player2) as matches from french_women_2013 group by player2;
select player, sum(matches) as totalmatches from french_women_matches group by player order by totalmatches desc limit 5;
+---------------------+--------------+
| player              | totalmatches |
+---------------------+--------------+
| Maria Sharapova     |            7 |
| Serena Williams     |            7 |
| Sara Errani         |            6 |
| Victoria Azarenka   |            6 |
| Agnieszka Radwanska |            5 |
+---------------------+--------------+

create table all_men_matches (player varchar(250), matches int);
insert into all_men_matches select * from us_men_matches union select * from french_men_matches union select * from aus_men_matches;
select player, sum(matches) as totalmatches from all_men_matches group by player order by totalmatches desc limit 5;
+-----------------+--------------+
| player          | totalmatches |
+-----------------+--------------+
| Roger Federer   |           15 |
| Rafael Nadal    |           14 |
| Richard Gasquet |           13 |
| David Ferrer    |           12 |
| Tomas Berdych   |           11 |
+-----------------+--------------+

create table all_women_matches (player varchar(250), matches int);
insert into all_women_matches select * from us_women_matches union select * from french_women_matches union select * from aus_women_matches;
select player, sum(matches) as totalmatches from all_women_matches group by player order by totalmatches desc limit 5;
+-----------------+--------------+
| player          | totalmatches |
+-----------------+--------------+
| Serena Williams |           11 |
| Maria Sharapova |           11 |
| Na Li           |            9 |
| Sloane Stephens |            8 |
| V Azarenka      |            7 |
+-----------------+--------------+

create table all_records_men like us_men_2013;
insert into all_records_men select * from us_men_2013 union select * from french_men_2013 union select * from aus_men_2013;

select player1, fsp_1 from all_records_men where fsp_1 = (select max(fsp_1) from all_records_men);
+--------------+-------+
| player1      | fsp_1 |
+--------------+-------+
| Rafael Nadal | 84    |
+--------------+-------+

select player2, fsp_2 from all_records_men where fsp_2 = (select max(fsp_2) from all_records_men);
+----------------+-------+
| player2        | fsp_2 |
+----------------+-------+
| Gael Monfils   | 84    |
| Carlos Berlocq | 84    |
+----------------+-------+

create table all_records_women like us_women_2013;
insert into all_records_women select * from us_women_2013 union select * from french_women_2013 union select * from aus_women_2013;

select player1, fsp_1 from all_records_women where fsp_1 = (select max(fsp_1) from all_records_women);
+-------------------------+-------+
| player1                 | fsp_1 |
+-------------------------+-------+
| Anabel Medina Garrigues | 86    |
+-------------------------+-------+

select player2, fsp_2 from all_records_women where fsp_2 = (select max(fsp_2) from all_records_women);
+----------+-------+
| player2  | fsp_2 |
+----------+-------+
| S Errani | 93    |
+----------+-------+

select player1, sum(result) as wins, sum(ufe_1)/sum(tpw_1) as ufeperc from all_records_men group by player1 order by wins desc limit 3;
+--------------------+------+---------------------+
| player1            | wins | ufeperc             |
+--------------------+------+---------------------+
| Stanislas Wawrinka |   12 | 0.20803629293583928 |
| Rafael Nadal       |   12 | 0.24724061810154527 |
| Novak Djokovic     |   10 | 0.16037735849056603 |
+--------------------+------+---------------------+

select player1, sum(result) as wins, sum(ufe_1)/sum(tpw_1) as ufeperc from all_records_women group by player1 order by wins desc limit 3;
+---------------------+------+--------------------+
| player1             | wins | ufeperc            |
+---------------------+------+--------------------+
| Serena Williams     |   10 | 0.2927170868347339 |
| Agnieszka Radwanska |    9 | 0.2047556142668428 |
| Na Li               |    7 | 0.4010416666666667 |
+---------------------+------+--------------------+

