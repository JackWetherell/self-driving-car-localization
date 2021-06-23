/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <math.h>
#include "helper_functions.h"
#include "particle_filter.h"


void ParticleFilter::init(double x, double y, double theta, double std[])
{
    // Check to see if the filter is already initialized 
    if(is_initialized == true)
    {
        return;
    }
    else
    {
        // Number of particles to generate
        num_particles = 100;

        // Creating Gaussian distributions
        std::normal_distribution<double> dist_x(x, std[0]);
        std::normal_distribution<double> dist_y(y, std[1]);
        std::normal_distribution<double> dist_theta(theta, std[2]);

        // Create random engine to use for sampling
        std::default_random_engine gen;

        // Generate particles
        for(int n = 0; n < num_particles; n++)
        {
            Particle particle;
            particle.id = n;
            particle.x = dist_x(gen);
            particle.y = dist_y(gen);
            particle.theta = dist_theta(gen);
            particle.weight = 1.0;
            this->particles.push_back(particle);
        }

        // The filter is now initialized.
        is_initialized = true;    
    }
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
    // Creating noGaussianrmal distributions
    std::normal_distribution<double> dist_x(0, std_pos[0]);
    std::normal_distribution<double> dist_y(0, std_pos[1]);
    std::normal_distribution<double> dist_theta(0, std_pos[2]);

    // Create cut-off for determining that the yaw is not changing to determine which equations of motion to use
    double cutoff = 1e-5;

    // Implement motion model
    for (int n = 0; n < num_particles; n++)
    {
        // Case where yaw is not changing
        if (fabs(yaw_rate) < cutoff)
        {
            particles[n].x += velocity * delta_t * cos(particles[n].theta);
            particles[n].y += velocity * delta_t * sin(particles[n].theta);
        }

        // Case where yaw is not changing
        else
        {
            particles[n].x += (velocity / yaw_rate) * (sin( particles[n].theta + yaw_rate * delta_t) - sin(particles[n].theta));
            particles[n].y += (velocity / yaw_rate) * (cos( particles[n].theta) - cos(particles[n].theta + yaw_rate * delta_t));
            particles[n].theta += yaw_rate * delta_t;
        }

        // Create random engine to use for sampling
        std::default_random_engine gen;

        // Adding noise from Gaussian distribution
        particles[n].x += dist_x(gen);
        particles[n].y += dist_y(gen);
        particles[n].theta += dist_theta(gen);
    }
}


void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations)
{
    for(short unsigned int n = 0; n < observations.size(); n++)
    {
        // Begin with large closest distance
        double closest_distance = 1e5;

        // Initialise observation id
        int id;

        for(short unsigned m = 0; m < predicted.size(); m++)
        {
            double x_distance = observations[n].x - predicted[m].x;
            double y_distance = observations[n].y - predicted[m].y;
            double distance = dist(0, 0, x_distance, y_distance);

            // Update if this is a closer distance
            if (distance < closest_distance)
            {
                id = predicted[m].id;
                closest_distance = distance; // Use as new closest distance
            }
        }

        // Chose observation based on id with closest distance
        observations[n].id = id;
    }
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
    for (int n = 0; n < num_particles; n++)
    {
        std::vector<LandmarkObs> in_range;
        for(long unsigned int m = 0; m < map_landmarks.landmark_list.size(); m++)
        {
            double distance_x = particles[n].x - map_landmarks.landmark_list[m].x_f;
            double distance_y = particles[n].y - map_landmarks.landmark_list[m].y_f;
            if(pow(distance_x, 2) + pow(distance_y, 2) <= pow(sensor_range, 2))
            {
                in_range.push_back(LandmarkObs{map_landmarks.landmark_list[m].id_i, map_landmarks.landmark_list[m].x_f, map_landmarks.landmark_list[m].y_f});
            }
        }

        // Transform using homogeneous transformation matrices
        std::vector<LandmarkObs> transformed;
        for(long unsigned int m = 0; m < observations.size(); m++)
        {
            double x_m = cos(particles[n].theta)*observations[m].x - sin(particles[n].theta)*observations[m].y + particles[n].x;
            double y_m = sin(particles[n].theta)*observations[m].x + cos(particles[n].theta)*observations[m].y + particles[n].y;
            transformed.push_back(LandmarkObs{observations[m].id, x_m, y_m});
        }

        // Perform data association
        dataAssociation(in_range, transformed);

        // Initialise weights
        particles[n].weight = 1.0;

        // Calculate weights
        for(long unsigned int m = 0; m < transformed.size(); m++)
        {
            long unsigned int count = 0;
            bool found = false;
            double landmark_x;
            double landmark_y;
            while(found == false && count < in_range.size())
            {
                if(in_range[count].id == transformed[m].id) 
                {
                    landmark_x = in_range[count].x;
                    landmark_y = in_range[count].y;
                    found = true;
                }
                count++;
            }

            // Calculate weight
            double dist_x = transformed[m].x - landmark_x;
            double dist_y = transformed[m].y - landmark_y;
            double weight = (1.0 / (2.0*M_PI*std_landmark[0]*std_landmark[1])) * exp(-(pow(dist_x, 2)/(2.0*pow(std_landmark[0], 2)) + (pow(dist_y, 2)/(2.0*pow(std_landmark[1], 2)))));
            if (weight == 0)
            {
                particles[n].weight *= 0.00001;
            } 
            else
            {
                particles[n].weight *= weight;
            }
      }
  }
}


void ParticleFilter::resample()
{
    // Get weights and max weight.
    std::vector<double> weights;
    double maxWeight = std::numeric_limits<double>::min();
    for(int i = 0; i < num_particles; i++)
    {
        weights.push_back(particles[i].weight);
        if(particles[i].weight > maxWeight)
        {
            maxWeight = particles[i].weight;
        }
    }

    // Creating distributions.
    std::uniform_real_distribution<double> distDouble(0.0, maxWeight);
    std::uniform_int_distribution<int> distInt(0, num_particles - 1);

    // Create random engine to use for sampling
    std::default_random_engine gen;

    // Generating index.
    int index = distInt(gen);
    double beta = 0.0;

    // The wheel
    std::vector<Particle> resampledParticles;
    for(int i = 0; i < num_particles; i++)
    {
        beta += distDouble(gen) * 2.0;
        while( beta > weights[index])
        {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        resampledParticles.push_back(particles[index]);
    }
    particles = resampledParticles;
}


void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}


std::string ParticleFilter::getAssociations(Particle best)
{
    std::vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1); 
    return s;
}


std::string ParticleFilter::getSenseCoord(Particle best, std::string coord)
{
    std::vector<double> v;
    if (coord == "X")
    {
        v = best.sense_x;
    }
    else
    {
        v = best.sense_y;
    }
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    std::string s = ss.str();
    s = s.substr(0, s.length()-1); 
    return s;
}