using System;
using Microsoft.ML.Probabilistic.Distributions;
using Microsoft.ML.Probabilistic.Models;

namespace AB_testing
{
	class MainClass
	{
		public static void Main(string[] args)
        {
            // read data from console
            string input;

            Console.WriteLine("*** SITE A ***");
            Console.Write("Total number of trials: ");
            input = Console.ReadLine();
            int siteATotal = Int32.Parse(input);
            Console.Write("Number of successes: ");
            input = Console.ReadLine();
            int numSiteASuccesses = Int32.Parse(input);

            Console.WriteLine("\n*** SITE B ***");
            Console.Write("Total number of trials: ");
            input = Console.ReadLine();
            int siteBTotal = Int32.Parse(input);
            Console.Write("Number of successes: ");
            input = Console.ReadLine();
            int numSiteBSuccesses = Int32.Parse(input);

            // compute frequentist statistics
            double siteAfreq = (double)numSiteASuccesses / (double)siteATotal;
            double siteBfreq = (double)numSiteBSuccesses / (double)siteBTotal;

            Console.WriteLine("\n=== Frequentist success rate estimates ===");
            Console.WriteLine("Site A success rate: " + siteAfreq);
            Console.WriteLine("Site B success rate: " + siteBfreq);

            // latent model variables
            var siteARate = Variable.Beta(1, 1);
            var siteBRate = Variable.Beta(1, 1);

            // observed variables
            var siteASucces = Variable.Binomial(siteATotal, siteARate);
            var siteBSucces = Variable.Binomial(siteBTotal, siteBRate);

            siteASucces.ObservedValue = numSiteASuccesses;
            siteBSucces.ObservedValue = numSiteBSuccesses;

			// inference
            InferenceEngine engine = new InferenceEngine();
            engine.ShowProgress = false;

            Beta siteARatePosterior = engine.Infer<Beta>(siteARate);
            Beta siteBRatePosterior = engine.Infer<Beta>(siteBRate);

            Console.WriteLine("\n=== Bayesian inference results ===");

            Console.WriteLine("Site A rate posterior = " + siteARatePosterior);
            Console.WriteLine("Site B rate posterior = " + siteBRatePosterior);

            // sample from marginal and compute probabilities
            int numSamples = 1000000;
            double diff = 0;
            double preferA = 0, preferB = 0;
            for (int i = 0; i < numSamples; i++)
            {
                diff = siteARatePosterior.Sample() - siteBRatePosterior.Sample();
                if (diff > 0)
                    preferA++;
                else
                    preferB++;
            }

            double probSiteAIsBetter = (double)(preferA / numSamples);
            double probSiteBIsBetter = 1 - probSiteAIsBetter;

            Console.WriteLine("\nP(Site A is better than B) =  " + probSiteAIsBetter);
            Console.WriteLine("P(Site B is better than A) =  " + probSiteBIsBetter);

            Console.WriteLine("Press any key ...");
            Console.ReadKey();
        }
	}
}
