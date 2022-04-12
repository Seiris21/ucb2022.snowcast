---
layout: default
---

# Predicting Snowpack in California’s Sierra Nevada Mountains

## Web Deliverable requirements
Requirements from Week 1 PPT:  

The final web deliverable is a chance to showcase your project,
not just for peers and instructors but also for a broader
audience. Think of it as a portfolio. 

At minimum, the web deliverable should: 
- describe the problem
- model results
- evaluation
- impact
- include a biography or group intro page.   
Interfaces that allow user interaction when appropriate are encouraged, for example, to explore different data sets or parameter choices.

### Problem
Estimation of snowfall in the Sierra Nevadas is critially important for the people of California and those who benefit from California agriculture. 75% of agricultural water supply across California are derived from precipitation and snow in the Sierra Nevadas.  
Current methods to estimate snow water equivalent (SWE) are either biased or expensive to conduct. Snow Telemetry (SNOTEL) sites are an automated network of snowpack and related climate sensors. Although there is constant data due to it's automated nature, the distribution of SNOTEL sites are biased towards lower elevations and do not accurately reflect all areas of interest. Airborne Snow Observatory (ASO) is a program to measure SWE through Light Detection and Ranging (lidar) and is the current gold standard for SWE estimations with it's ability to map entire regions through flyby missions. However ASO data collection is costly, and in recent years have gone private.  
Accurate estimates of snowfall would provide a valuable tool for natural resource managements, allowing for better usage recommendations and minimizing negative impacts for the agricultural region.



### [Biography](biography.md)

<div class="posts">
  {% for post in paginator.posts %}
    <article class="post">
      <a href="{{ site.baseurl }}{{ post.url }}">
        <h1>{{ post.title }}</h1>

        <div>
          <p class="post_date">{{ post.date | date: "%B %e, %Y" }}</p>
        </div>
      </a>
      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.baseurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}

  <!-- pagination -->
  {% if paginator.total_pages > 1 %}
  <div class="pagination">
    {% if paginator.previous_page %}
      <a href="{{ paginator.previous_page_path | prepend: site.baseurl | replace: '//', '/' }}">&laquo; Prev</a>
    {% else %}
      <span>&laquo; Prev</span>
    {% endif %}

    {% for page in (1..paginator.total_pages) %}
      {% if page == paginator.page %}
        <span class="webjeda">{{ page }}</span>
      {% elsif page == 1 %}
        <a href="{{ '/' | prepend: site.baseurl | replace: '//', '/' }}">{{ page }}</a>
      {% else %}
        <a href="{{ site.paginate_path | prepend: site.baseurl | replace: '//', '/' | replace: ':num', page }}">{{ page }}</a>
      {% endif %}
    {% endfor %}

    {% if paginator.next_page %}
      <a href="{{ paginator.next_page_path | prepend: site.baseurl | replace: '//', '/' }}">Next &raquo;</a>
    {% else %}
      <span>Next &raquo;</span>
    {% endif %}
  </div>
  {% endif %}
</div>
