#!/usr/bin/perl

use strict;
use warnings;

$ENV{PGSERVICEFILE} = '/home/nobody/.pg_service_backprice.conf';

use BOM::MarketData   qw(create_underlying);
use VolSurface::Utils qw(get_strike_for_spot_delta);
use BOM::Config::Chronicle;
use Quant::Framework::VolSurface::Delta;
use Quant::Framework::VolSurface::Moneyness;
use Date::Utility;
use File::Slurp;
use List::Util qw(min sum);
use Time::HiRes qw(usleep ualarm gettimeofday tv_interval);
use Data::Dump 'pp';
use feature 'say';

=head1 NAME

get_vol_smile

=head1 DESCRIPTION

Returns volatilities when specified a tenor, strike, from_date and asset

=cut

# Set start date of contract
my $from_date = Date::Utility->new("2023-02-06 16:00:00");

# Set underlyings
my $ccy = "frxEURUSD";
my $stock = "OTC_FTSE";

# Create underlyings
my $underlying_ccy   = create_underlying($ccy, $from_date);
my $underlying_stock = create_underlying($stock, $from_date);

# Create exchange rate volatility surface
my $volsurface_ccy = Quant::Framework::VolSurface::Delta->new(
    underlying       => $underlying_ccy,
    for_date         => $from_date,
    chronicle_reader => BOM::Config::Chronicle::get_chronicle_reader(1),
);

# Create stock volatility surface
my $volsurface_stock = Quant::Framework::VolSurface::Moneyness->new(
    underlying       => $underlying_stock,
    for_date         => $from_date,
    chronicle_reader => BOM::Config::Chronicle::get_chronicle_reader(1),
);

# Further arguments
my $tenor = 1;
my $strike = 1;
my $moneyness = 1.0 * 100;

my $volatility_ccy = $volsurface_ccy->get_volatility({strike => $strike, from => $from_date, to => $from_date->plus_time_interval($tenor.'d')});
my $volatility_stock = $volsurface_stock->get_volatility({moneyness => $moneyness, from => $from_date, to => $from_date->plus_time_interval($tenor.'d')});

print($ccy." volatility is ".$volatility_ccy."\n");
print($stock." volatility is ".$volatility_stock."\n");