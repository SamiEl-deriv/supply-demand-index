#!/usr/bin/perl
use strict;
use warnings;
$ENV{PGSERVICEFILE} = '/home/nobody/.pg_service_backprice.conf';
# use Volatility::EconomicEvents;

use Date::Utility;
use BOM::MarketData   qw(create_underlying);
use BOM::Product::ContractFactory qw(produce_contract);
use Pricing::Engine::Markup::IntradayMeanReversionMarkup;
use Text::CSV::Slurp;
use Quant::Framework::EconomicEventCalendar;
use BOM::Config::Chronicle;
use List::Util qw(uniq);
use File::Slurp qw(append_file write_file);
use List::Util qw(max min sum first);
use Data::Dump qw(pp);

my $cr = BOM::Config::Chronicle::get_chronicle_reader(1);
my $cw = BOM::Config::Chronicle::get_chronicle_writer();

# UPORDOWN_FRXAUDJPY_4000.00_1710494847_1710806399_123000000_80000000
my $underlying_name = "FRXAUDJPY";
my $from_raw = 1710494847;
my $from = Date::Utility->new($from_raw);
my $to_raw = 1710806399;
my $to = Date::Utility->new($to_raw);
my $option_type = "UPORDOWN";
my $stake = 4000.00;
my $up_barr = "123000000";
my $down_barr = "80000000";
my $short_code = join("_", $option_type, $underlying_name, $stake, $from->epoch, $to->epoch, $up_barr, $down_barr);

# my $short_code = "UPORDOWN_FRXAUDJPY_4000.00_1710494847_1710806399_123000000_80000000"

my $symbol = "USD";
my $c = produce_contract($short_code,$symbol);

my $p = $c->build_parameters;
$p->{date_pricing} = $c->date_start;

$c = produce_contract($p);
my %args = (
        symbol => $symbol,
        for_date => $from,
        chronicle_reader => $cr,
        chronicle_writer => $cw,
    );

my $implied = Quant::Framework::ImpliedRate->new(
    symbol           => $implied_symbol,
    rates            => $implied_rates,
    recorded_date    => Date::Utility->new,
    chronicle_reader => BOM::Config::Chronicle::get_chronicle_reader(),
    chronicle_writer => BOM::Config::Chronicle::get_chronicle_writer(),
);
my $currency = Quant::Framework::Currency->new(%args);
my $underlying = create_underlying($underlying_name, $from);
my $volsurface = Quant::Framework::VolSurface::Delta->new(
    underlying       => $underlying,
    for_date         => $from,
    chronicle_reader => $cr,
);

my $variance_table = $volsurface->variance_table;
my $var_table_expiries = $volsurface->_sorted_variance_table_expiries;

my $duration_days = ($to_raw - $from_raw) / 86400 / 365;
print 'shortcode: ' .$short_code ."\n";
print 'pricing vol: ' . $c->pricing_vol . "\n";
print 'get_volatility atm vol: ' . $volsurface->get_volatility({market=>'ATM', from => $from, to => $to}) . "\n";
print 'volsurface atm vol: ';
print pp $volsurface->get_smile($from_raw, $to_raw);

print "\n" . 'volsurface (ON): ';
print pp $volsurface->surface_data->{1}->{smile};
print "\n" . 'get_smile for ON: ';
print pp $volsurface->get_smile($from_raw, $from_raw + 86400);

print "\n" . 'volsurface (1w): ';
print pp $volsurface->surface_data->{7}->{smile};
print "\n" . 'get_smile for 1w: ';
print pp $volsurface->get_smile($from_raw, $from_raw + 86400 * 7);
print "\n" . 'Variance for ON: ';
print pp $variance_table->{$var_table_expiries->[1]};

print "\n" . 'ON Contract Duration (days): ';
print "\n" . $duration_days;

print "$symbol interest rate: " . $currency->interest->rate_for($duration_days / 360) . "\n";
