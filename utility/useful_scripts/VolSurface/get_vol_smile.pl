#!/usr/bin/perl

use strict;
use warnings;

$ENV{PGSERVICEFILE} = '/home/nobody/.pg_service_backprice.conf';

use BOM::MarketData   qw(create_underlying);
use VolSurface::Utils qw(get_strike_for_spot_delta);
use BOM::Config::Chronicle;
use Quant::Framework::VolSurface::Delta;
use Date::Utility;
use File::Slurp;
use List::Util qw(min sum);
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Data::Dump 'pp';
use feature 'say';

=head1 NAME

get_vol_smile

=head1 DESCRIPTION

Returns a csv containing the volsmiles, market and calculated for each specified currency, start and tenor.

To use, modify @from_raw, @underlying_names.

Generates output_smile.csv in the same folder containing

underlying - The underlying exchange rate (This implementation supports only currencies at the moment)
tenor - The tenor (time to maturity/expiry) of the contract
start_date - Start date of contract (Actual start date may depend on cut-off times)
premium_adj - 1 if premium-adjusted, 0 if not
market_25, market_50, market_75 - market smile for forex
calc_25, calc_50, calc_75 - weight-adjusted market smile for forex

=cut


# Edit location of output
my $output_file = './output_smile.csv';

# market vols refer to unadulterated bloomberg vols. calc_vols refer to after we apply weights
write_file($output_file, "underlying,tenor,start_date,premium_adj,market_25,market_50,market_75,calc_25,calc_50,calc_75\n");  # clear it


# any historical data, modify for starting date
my @from_raw = sort ("2022-09-06 10:00:00","2022-09-01 10:00:00","2022-07-28 10:00:00","2022-06-04 10:00:00");
my @from = map {Date::Utility->new($_)} @from_raw;

my @minor_pairs = ("frxAUDCAD", "frxAUDCHF","frxAUDNZD","frxEURNZD", "frxGBPCAD","frxGBPCHF","frxGBPNZD","frxNZDJPY","frxNZDUSD","frxUSDMXN","frxUSDPLN");
my @major_pairs = ("frxAUDJPY","frxAUDUSD", "frxEURAUD","frxEURCAD", "frxEURCHF","frxEURGBP","frxEURJPY","frxEURUSD","frxGBPAUD","frxGBPUSD","frxGBPJPY","frxUSDCAD","frxUSDCHF","frxUSDJPY");

# Add your own list or use the above
my @underlying_names = @major_pairs;

# Main loop
foreach my $from_date (@from) {
    print "starting date is ".($from_date->datetime)."\n";
    foreach my $underlying_name (@underlying_names) {
        my $starting_time = gettimeofday();
        my $underlying = create_underlying($underlying_name, $from_date);
        my $volsurface = Quant::Framework::VolSurface::Delta->new(
            underlying       => $underlying,
            for_date         => $from_date,
            chronicle_reader => BOM::Config::Chronicle::get_chronicle_reader(1),
        );

        my $tenors = $volsurface->original_term_for_smile;
        my $premium_adjusted = $underlying->market_convention->{delta_premium_adjusted};
        #my @expiries         = map { Date::Utility->new($_) } @{$volsurface->_sorted_variance_table_expiries};
        #my $tenors           = $volsurface->original_term_for_smile;
        my $spot = $underlying->tick_at($from_date->epoch)->quote;
        my $step_size = $underlying->pip_size * 100;;

        print "underlying is ".$underlying_name."\n";
        my @tenor_elapsed = ();
        foreach my $tte (@$tenors) {
            my $starting_time = gettimeofday();
            my $smile = $volsurface->surface_data->{$tte}->{smile};

            # Each smile volatility corresponds to the following 3 strings. Avoid using 25, 50, 75 unless accessing the raw smile ($smile)
            my @market_deltas = ("25C", "ATM", "25P");
            my @calc_vols = map {$volsurface->get_volatility({market => $_, from => $from_date, to => $from_date->plus_time_interval($tte.'d')})} @market_deltas;
            append_file($output_file, "$underlying_name,$tte,".$from_date->epoch.",$premium_adjusted,".$smile->{25}.",".$smile->{50}.",".$smile->{75}.",".$calc_vols[0].",".$calc_vols[1].",".$calc_vols[2]."\n");
            my $time_elapsed = gettimeofday()-$starting_time;
            print("Elapsed time for tenor $tte: $time_elapsed seconds\n");
            push(@tenor_elapsed, $time_elapsed);
        }
        print("Total Elapsed time for $underlying_name: ".(sum (@tenor_elapsed))." seconds\n\n");
    }
}

