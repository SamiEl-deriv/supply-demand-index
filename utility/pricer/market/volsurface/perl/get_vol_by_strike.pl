#!/usr/bin/perl

use strict;
use warnings;

$ENV{PGSERVICEFILE} = '/home/nobody/.pg_service_backprice.conf';

use BOM::MarketData   qw(create_underlying);
use VolSurface::Utils qw(get_strike_for_spot_delta get_delta_for_strike);
use BOM::Config::Chronicle;
use Quant::Framework::VolSurface::Delta;
use Date::Utility;
use File::Slurp;
use List::Util qw(min sum);
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);

# Edit location of output
my $output_file = './output_delta.csv';
write_file($output_file, "");  # clear it

# append_file($output_file, "underlying,vol,strike,start_date,end_date,interest_rate,dividend_rate,spot\n");
my @buffer = ("underlying,vol,strike,delta,start_date,end_date,interest_rate,dividend_rate,spot,premium_adj\n");

# any historical data
my @from_raw = sort ("2022-09-06 10:00:00");
# ,"2022-09-01 10:00:00","2022-07-28 10:00:00","2022-06-04 10:00:00");
my @from = map {Date::Utility->new($_)} @from_raw;

my @minor_pairs = ("frxAUDCAD", "frxAUDCHF","frxAUDNZD","frxEURNZD", "frxGBPCAD","frxGBPCHF","frxGBPNZD","frxNZDJPY","frxNZDUSD","frxUSDMXN","frxUSDPLN");
my @major_pairs = ("frxAUDJPY","frxAUDUSD", "frxEURAUD","frxEURCAD", "frxEURCHF","frxEURGBP","frxEURJPY","frxEURUSD","frxGBPAUD","frxGBPUSD","frxGBPJPY","frxUSDCAD","frxUSDCHF","frxUSDJPY");
my @underlying_names = ("frxAUDJPY","frxEURUSD");

foreach my $from_date (@from) {
    print "starting date is".($from_date->datetime)."\n";
    foreach my $underlying_name (@underlying_names) {
        my $underlying = create_underlying($underlying_name, $from_date);
        my $volsurface = Quant::Framework::VolSurface::Delta->new(
            underlying       => $underlying,
            for_date         => $from_date,
            chronicle_reader => BOM::Config::Chronicle::get_chronicle_reader(1),
        );

        my @tenors = map {$_.'d'} sort {$a <=> $b} keys %{$volsurface->surface_data};
        my $premium_adjusted = $underlying->market_convention->{delta_premium_adjusted};
        #my @expiries         = map { Date::Utility->new($_) } @{$volsurface->_sorted_variance_table_expiries};
        #my $tenors           = $volsurface->original_term_for_smile;
        my $spot = $underlying->tick_at($from_date->epoch)->quote;
        my $step_size = $underlying->pip_size * 100;;

        my @to = map {$from_date->plus_time_interval($_)} @tenors;

        print "underlying is ".$underlying_name."\n";
        my @tenor_elapsed = ();

        foreach my $to_date (@to) {
            my $starting_time = gettimeofday();
            my $t = ($to_date->epoch - $from_date->epoch) / (365 * 86400);
            print "tenor is ".($t*365)."\n";

            my $r = $underlying->interest_rate_for($t);
            my $q = $underlying->dividend_rate_for($t);
            my $i = 0;
            # delta point, currently it is 25,50,75 delta.
            for (my $strike = $step_size; $strike <= (1.5+$i*0.025)*$spot; $strike += $step_size) {
                my $vol = $volsurface->get_volatility({strike => $strike, from => $from_date, to => $to_date});
                # my $delta = get_delta_for_strike({t => $t, spot => $spot, premium_adjusted => 0, r_rate => $r, q_rate=>$q, atm_vol =>});
                # $new_args{atm_vol} ||= $self->get_volatility({
                #     delta => 50,
                #     from  => $args->{from},
                #     to    => $args->{to},
                # });
                # append_file($output_file, "$underlying_name,$vol,$strike," . $from_date->epoch . "," . $to_date->epoch . ",$r,$q,$spot,$premium_adjusted\n");
                my $line = "$underlying_name,$vol,$strike," . $from_date->epoch . "," . $to_date->epoch . ",$r,$q,$spot,$premium_adjusted\n";
                push(@buffer, $line); 
                $i += 1;
            }
            append_file($output_file, \@buffer);
            @buffer = ();
            my $time_elapsed = gettimeofday()-$starting_time;
            print("Elapsed time for tenor ".($t*365).": $time_elapsed seconds\n");
            push(@tenor_elapsed, $time_elapsed);
        }
        print("Total Elapsed time for $underlying_name: ".(sum (@tenor_elapsed))." seconds\n");
    }
}